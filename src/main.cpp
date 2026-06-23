#include "agent.hpp"
#include "config.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <csignal>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <chrono>
#include <readline/readline.h>
#include <readline/history.h>

namespace fs = std::filesystem;

static volatile sig_atomic_t g_interrupted = 0;
static void handle_signal(int) { g_interrupted = 1; }

// ── Spinner ───────────────────────────────────────────────────────────────────
class Spinner {
    std::thread         th_;
    std::atomic<bool>   running_{false};
public:
    void start() {
        stop();   // safe restart
        running_ = true;
        th_ = std::thread([this] {
            static const char* F[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
            int i = 0;
            while (running_) {
                fprintf(stderr, "\r\033[2m%s\033[0m ", F[i++ % 10]);
                fflush(stderr);
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
            fprintf(stderr, "\r\033[K");
            fflush(stderr);
        });
    }
    void stop() {
        if (!running_) return;
        running_ = false;
        if (th_.joinable()) th_.join();
    }
    bool is_running() const { return running_.load(); }
    ~Spinner() { stop(); }
};

static void print_usage(const char* prog) {
    std::cerr
        << "Использование: " << prog
        << " [--config <путь>] [--mode plan|build] [--verbose] [--no-stream]\n"
        << "  --mode plan    только анализ и планирование (без инструментов)\n"
        << "  --mode build   выполнение задач с инструментами (по умолчанию)\n"
        << "Команды в чате: /plan  /build  /reset  /info  /exit\n";
}

static std::string prompt_working_dir(const std::string& current) {
    const std::string def = (current.empty() || current == ".")
        ? (std::string(getenv("HOME") ? getenv("HOME") : "/home") + "/Projects")
        : current;
    std::cout << "Рабочая папка [" << def << "]: " << std::flush;
    std::string input;
    std::getline(std::cin, input);
    return input.empty() ? def : input;
}

static void print_mode_banner(AgentMode m) {
    if (m == AgentMode::Plan)
        std::cout << "  \033[33m◆ план\033[0m  \033[2mтолько чтение · без изменений\033[0m\n";
    else
        std::cout << "  \033[32m◆ build\033[0m \033[2mполный доступ к инструментам\033[0m\n";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);

    std::string config_path        = "./config.json";
    bool        override_verbose    = false;
    bool        override_no_stream  = false;
    bool        mode_override       = false;
    AgentMode   override_mode       = AgentMode::Build;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            override_mode  = mode_from_string(argv[++i]);
            mode_override  = true;
        } else if (arg == "--verbose") {
            override_verbose = true;
        } else if (arg == "--no-stream") {
            override_no_stream = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Неизвестный аргумент: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    AppConfig cfg;
    try {
        cfg = load_config(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[ошибка] " << e.what() << "\n";
        return 1;
    }

    if (override_verbose)   cfg.agent.verbose = true;
    if (override_no_stream) cfg.agent.stream  = false;
    if (mode_override)      cfg.agent.mode    = override_mode;

    if (cfg.tools.enabled) {
        cfg.tools.working_dir = prompt_working_dir(cfg.tools.working_dir);
        std::error_code ec;
        fs::create_directories(cfg.tools.working_dir, ec);
        if (ec) {
            std::cerr << "[ошибка] Не удалось создать папку '"
                      << cfg.tools.working_dir << "': " << ec.message() << "\n";
            return 1;
        }
        std::cout << "Папка готова: " << fs::absolute(cfg.tools.working_dir) << "\n\n";
    }

    std::unique_ptr<Agent> agent;
    try {
        std::cerr << "[инфо] Загрузка модели: " << cfg.model.path << " ...\n";
        agent = std::make_unique<Agent>(cfg);
        std::cerr << "[инфо] Модель загружена.\n\n";
    } catch (const std::exception& e) {
        std::cerr << "[ошибка] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n";
    print_mode_banner(agent->mode());
    std::cout << "  \033[2m/plan  /build  /reset  /verbose  /exit\033[0m\n\n";

    bool verbose_tools = false;

    while (!g_interrupted) {
        const bool is_plan     = agent->mode() == AgentMode::Plan;
        const char* mode_color = is_plan ? "\033[33m" : "\033[32m";
        const char* mode_sym   = is_plan ? "◆" : "◆";
        std::string prompt_str = std::string(mode_color) + mode_sym + "\033[0m \033[1m>\033[0m ";

        char* raw = readline(prompt_str.c_str());
        if (!raw) break;                    // Ctrl+D — выход
        std::string line(raw);
        if (!line.empty()) add_history(raw);
        free(raw);

        if (line.empty()) continue;

        if (line == "/exit" || line == "/quit") break;

        if (line == "/plan") {
            agent->set_mode(AgentMode::Plan);
            print_mode_banner(AgentMode::Plan);
            continue;
        }
        if (line == "/build") {
            agent->set_mode(AgentMode::Build);
            print_mode_banner(AgentMode::Build);
            continue;
        }
        if (line == "/reset") {
            agent->reset_history();
            std::cout << "[история очищена]\n";
            continue;
        }
        if (line == "/info") {
            agent->print_info();
            continue;
        }
        if (line == "/verbose") {
            verbose_tools = !verbose_tools;
            std::cout << (verbose_tools ? "[verbose: ON — показываю сырые tool-блоки]\n"
                                        : "[verbose: OFF]\n");
            continue;
        }

        try {
            if (cfg.agent.stream) {
                Spinner spinner;
                bool header_shown = false;

                auto stream_cb = [&](const std::string& piece) -> bool {
                    spinner.stop();   // stop on any output
                    if (!header_shown) {
                        std::cout << "\033[2m  ╰─\033[0m " << std::flush;
                        header_shown = true;
                    }
                    (void)verbose_tools;
                    std::cout << piece << std::flush;
                    return !g_interrupted; // sig_atomic_t is implicitly convertible to bool
                };

                auto think_cb = [&]() {
                    spinner.start();  // restart between tool calls
                };

                agent->chat(line, stream_cb, think_cb);
                spinner.stop();
                std::cout << "\n\n";
            } else {
                std::cout << agent->chat(line) << "\n\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "\n\033[31m  ✗ " << e.what() << "\033[0m\n";
        }

        if (g_interrupted) {
            std::cout << "\n[прервано]\n";
            g_interrupted = 0;
        }
    }

    std::cout << "\nПока.\n";
    return 0;
}
