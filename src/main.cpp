#include "agent.hpp"
#include "config.hpp"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <csignal>
#include <cstdlib>
#include <ctime>
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
        stop();
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
    ~Spinner() { stop(); }
};

// ── Utilities ─────────────────────────────────────────────────────────────────
static std::string expand_home(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = getenv("HOME");
    return home ? std::string(home) + path.substr(1) : path;
}

static std::string make_session_slug(const std::string& msg) {
    std::string slug;
    bool last_dash = false;
    for (unsigned char c : msg) {
        if (c >= 0x80 || std::isalnum(c)) {
            slug += (char)c;
            last_dash = false;
        } else if (std::isspace(c) || c == '-' || c == '_') {
            if (!last_dash && !slug.empty()) {
                slug += '-';
                last_dash = true;
            }
        }
        if (slug.size() >= 40) break;
    }
    while (!slug.empty() && slug.back() == '-') slug.pop_back();
    return slug.empty() ? "session" : slug;
}

static std::string unique_session_path(const std::string& dir, const std::string& slug) {
    const std::string base = dir + "/" + slug;
    std::string path = base + ".json";
    for (int i = 2; fs::exists(path); ++i)
        path = base + "_" + std::to_string(i) + ".json";
    return path;
}

// ── Session listing ───────────────────────────────────────────────────────────
struct SessionEntry {
    std::string        path;
    std::string        title;
    int                count = 0;
    fs::file_time_type mtime;
};

static std::vector<SessionEntry> scan_sessions(const std::string& dir_raw) {
    const std::string dir = expand_home(dir_raw);
    std::vector<SessionEntry> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (ec) { ec.clear(); continue; }
        if (e.path().extension() != ".json") continue;

        std::ifstream f(e.path());
        if (!f.is_open()) continue;
        nlohmann::json j;
        try { j = nlohmann::json::parse(f); } catch (...) { continue; }
        if (!j.is_array() || j.empty()) continue;

        SessionEntry se;
        se.path  = e.path().string();
        se.count = (int)j.size();
        std::error_code mec;
        se.mtime = e.last_write_time(mec);

        for (const auto& msg : j) {
            if (msg.value("role", "") == "user") {
                std::string t = msg.value("content", "");
                while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
                se.title = t.size() > 55 ? t.substr(0, 52) + "\xe2\x80\xa6" : t;
                break;
            }
        }
        if (se.title.empty()) se.title = e.path().stem().string();
        out.push_back(std::move(se));
    }
    std::sort(out.begin(), out.end(), [](const SessionEntry& a, const SessionEntry& b) {
        return a.mtime > b.mtime;
    });
    return out;
}

static int prompt_session(const std::vector<SessionEntry>& sessions) {
    std::cout << "\n  \033[1mСессии:\033[0m\n";
    std::cout << "    \033[2m[0]  новая сессия\033[0m\n";
    for (int i = 0; i < (int)sessions.size(); ++i)
        printf("    \033[36m[%d]\033[0m  %-56s \033[2m(%d сообщ.)\033[0m\n",
               i + 1, sessions[i].title.c_str(), sessions[i].count);
    printf("\n  Выбери [0-%d] или Enter для новой: ", (int)sessions.size());
    fflush(stdout);

    std::string line;
    if (!std::getline(std::cin, line)) return -1;
    if (line.empty()) return 0;
    try {
        const int n = std::stoi(line);
        if (n >= 0 && n <= (int)sessions.size()) return n;
    } catch (...) {}
    return 0;
}

// ── Session delete ────────────────────────────────────────────────────────────
// Shows numbered list, lets user pick one or more (comma-separated) to delete.
// Returns true if the current session was among deleted files.
static bool cmd_delete(const std::string& sessions_dir,
                       const std::string& current_path) {
    if (sessions_dir.empty()) {
        std::cout << "[sessions_dir не задан в config.json]\n";
        return false;
    }
    const auto list = scan_sessions(sessions_dir);
    if (list.empty()) {
        std::cout << "[нет сохранённых сессий]\n";
        return false;
    }

    std::cout << "\n  \033[1mУдалить сессию:\033[0m\n";
    for (int i = 0; i < (int)list.size(); ++i) {
        const bool cur = (list[i].path == current_path);
        printf("    \033[36m[%d]\033[0m  %-54s \033[2m(%d сообщ.)%s\033[0m\n",
               i + 1, list[i].title.c_str(), list[i].count,
               cur ? "  ← текущая" : "");
    }
    std::cout << "\n  Введи номера через запятую (или Enter для отмены): " << std::flush;

    std::string line;
    if (!std::getline(std::cin, line) || line.empty()) {
        std::cout << "[отмена]\n";
        return false;
    }

    bool deleted_current = false;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim whitespace
        const size_t s = tok.find_first_not_of(" \t");
        const size_t e = tok.find_last_not_of(" \t");
        if (s == std::string::npos) continue;
        tok = tok.substr(s, e - s + 1);
        try {
            const int n = std::stoi(tok);
            if (n < 1 || n > (int)list.size()) {
                std::cout << "  [пропущен неверный номер " << n << "]\n";
                continue;
            }
            const auto& se = list[n - 1];
            std::error_code ec;
            fs::remove(se.path, ec);
            if (ec)
                printf("  \033[31m✗\033[0m %s: %s\n", se.title.c_str(), ec.message().c_str());
            else {
                printf("  \033[32m✓\033[0m удалена «%s»\n", se.title.c_str());
                if (se.path == current_path) deleted_current = true;
            }
        } catch (...) {
            std::cout << "  [пропущен: «" << tok << "» не число]\n";
        }
    }
    return deleted_current;
}

// ── Banners ───────────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::cerr
        << "Использование: " << prog
        << " [--config <путь>] [--mode plan|build] [--verbose] [--no-stream]\n"
        << "Команды: /plan  /build  /reset  /sessions  /delete  /verbose  /exit\n";
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

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);

    std::string config_path        = "./config.json";
    bool        override_verbose   = false;
    bool        override_no_stream = false;
    bool        mode_override      = false;
    AgentMode   override_mode      = AgentMode::Build;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            override_mode = mode_from_string(argv[++i]);
            mode_override = true;
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

    if (override_verbose)    cfg.agent.verbose = true;
    if (override_no_stream)  cfg.agent.stream  = false;
    if (mode_override)       cfg.agent.mode    = override_mode;

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
        std::cerr << "[инфо] Модель загружена.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ошибка] " << e.what() << "\n";
        return 1;
    }

    // ── Session selection ─────────────────────────────────────────────────────
    const std::string sessions_dir_exp = expand_home(cfg.agent.sessions_dir);
    bool session_named = false;

    if (!cfg.agent.sessions_dir.empty()) {
        std::error_code ec;
        fs::create_directories(sessions_dir_exp, ec);

        const auto sessions = scan_sessions(cfg.agent.sessions_dir);
        if (!sessions.empty()) {
            const int choice = prompt_session(sessions);
            if (choice > 0 && choice <= (int)sessions.size()) {
                const auto& se = sessions[choice - 1];
                agent->set_session_path(se.path);
                const int loaded = agent->load_session(se.path);
                session_named = true;
                std::cout << "\n  \033[2m\xe2\x86\x91 загружена «" << se.title
                          << "» (" << loaded << " сообщ.)\033[0m\n";
            }
        }
    }

    std::cout << "\n";
    print_mode_banner(agent->mode());
    std::cout << "  \033[2m/plan  /build  /reset  /sessions  /delete  /verbose  /exit"
                 "  \xe2\x94\x82  Ctrl+C \xe2\x80\x94 прервать генерацию\033[0m\n\n";

    bool verbose_tools = false;

    while (!g_interrupted) {
        const bool is_plan = agent->mode() == AgentMode::Plan;
        std::string prompt_str = std::string(is_plan ? "\033[33m" : "\033[32m")
                                 + "\xe2\x97\x86\033[0m \033[1m>\033[0m ";

        char* raw = readline(prompt_str.c_str());
        if (!raw) break;   // Ctrl+D
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
            agent->set_session_path("");
            session_named = false;
            std::cout << "[история очищена \xe2\x80\x94 следующий запрос начнёт новую сессию]\n";
            continue;
        }
        if (line == "/sessions") {
            const std::string cur = agent->current_session_path();
            if (cur.empty())
                std::cout << "[сессия не сохранена \xe2\x80\x94 введите первый запрос]\n";
            else
                std::cout << "[текущая сессия: " << cur << "]\n";
            if (!cfg.agent.sessions_dir.empty()) {
                const auto list = scan_sessions(cfg.agent.sessions_dir);
                if (list.empty())
                    std::cout << "[нет сохранённых сессий]\n";
                for (int i = 0; i < (int)list.size(); ++i)
                    printf("  [%d]  %s  (%d сообщ.)%s\n",
                           i + 1, list[i].title.c_str(), list[i].count,
                           list[i].path == cur ? "  \xe2\x86\x90 текущая" : "");
            }
            continue;
        }
        if (line == "/delete") {
            const bool deleted_cur = cmd_delete(
                cfg.agent.sessions_dir, agent->current_session_path());
            if (deleted_cur) {
                agent->reset_history();
                agent->set_session_path("");
                session_named = false;
                std::cout << "[текущая сессия удалена \xe2\x80\x94 история очищена]\n";
            }
            continue;
        }
        if (line == "/info") { agent->print_info(); continue; }
        if (line == "/verbose") {
            verbose_tools = !verbose_tools;
            std::cout << (verbose_tools ? "[verbose: ON]\n" : "[verbose: OFF]\n");
            continue;
        }

        // ── Assign session path on first real message of a new session ────
        if (!session_named && !cfg.agent.sessions_dir.empty()) {
            const std::string slug = make_session_slug(line);
            const std::string path = unique_session_path(sessions_dir_exp, slug);
            agent->set_session_path(path);
            session_named = true;
        }

        try {
            if (cfg.agent.stream) {
                Spinner spinner;
                bool header_shown = false;

                auto stream_cb = [&](const std::string& piece) -> bool {
                    spinner.stop();
                    if (!header_shown) {
                        std::cout << "\033[2m  \xe2\x95\xb0\xe2\x94\x80\033[0m " << std::flush;
                        header_shown = true;
                    }
                    (void)verbose_tools;
                    std::cout << piece << std::flush;
                    return !g_interrupted;
                };

                auto think_cb = [&]() { spinner.start(); };

                agent->chat(line, stream_cb, think_cb);
                spinner.stop();
                std::cout << "\n\n";
            } else {
                std::cout << agent->chat(line) << "\n\n";
            }
            agent->save_history();
        } catch (const std::exception& e) {
            std::cerr << "\n\033[31m  \xe2\x9c\x97 " << e.what() << "\033[0m\n";
        }

        if (g_interrupted) {
            std::cout << "\n\033[2m[прервано]\033[0m\n\n";
            g_interrupted = 0;
        }
    }

    std::cout << "\n\xd0\x9f\xd0\xbe\xd0\xba\xd0\xb0.\n";
    return 0;
}
