#include "config.hpp"
#include "agent.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

static const char* RED    = "\033[31m";
static const char* YEL    = "\033[33m";
static const char* GRN    = "\033[32m";
static const char* DIM    = "\033[2m";
static const char* RESET  = "\033[0m";

static int g_pass = 0, g_fail = 0;

static std::string run_task(Agent& agent, const std::string& task) {
    std::cout << "\n" << YEL << ">>> " << task << RESET << "\n";
    std::string full;
    agent.chat(task, [&](const std::string& piece) -> bool {
        std::cout << piece;
        std::cout.flush();
        full += piece;
        return true;
    });
    std::cout << "\n";
    return full;
}

static void check(const std::string& label, bool ok) {
    if (ok) { std::cout << GRN << "  PASS" << RESET << "  " << label << "\n"; ++g_pass; }
    else     { std::cout << RED << "  FAIL" << RESET << "  " << label << "\n"; ++g_fail; }
}

int main() {
    const char* cfg_path = "/home/kolpachkov/Projects/AI agent/config.json";
    AppConfig cfg = load_config(cfg_path);
    cfg.inference.max_tokens = 512;
    cfg.model.n_gpu_layers = 0;   // CPU-only so test doesn't fight main agent for VRAM
    cfg.tools.working_dir = "/tmp";

    // ── Test 1: basic answer, no tool calls ──────────────────────────────────
    {
        Agent a(cfg);
        auto r = run_task(a, "скажи только слово: яблоко");
        check("ответил (не пустой)", !r.empty());
        check("нет JSON {", r.find("{\"output\"") == std::string::npos);
    }

    // ── Test 2: filesystem tool — read existing dir ──────────────────────────
    {
        cfg.tools.working_dir = "/home/kolpachkov/Projects/AI agent";
        Agent a(cfg);
        auto r = run_task(a, "выведи список файлов в директории src/");
        check("использовал list_dir/tree", r.find("main.cpp") != std::string::npos
                                         || r.find("agent.cpp") != std::string::npos);
        check("нет JSON {", r.find("{\"output\"") == std::string::npos);
    }

    // ── Test 3: write + read file ────────────────────────────────────────────
    {
        cfg.tools.working_dir = "/tmp";
        Agent a(cfg);
        ::remove("/tmp/ai_test_123.txt");
        auto r = run_task(a, "создай файл /tmp/ai_test_123.txt со строкой 'test_ok' и прочитай его");
        check("файл создан", ::system("test -f /tmp/ai_test_123.txt") == 0);
        check("содержит test_ok", r.find("test_ok") != std::string::npos);
    }

    // ── Test 4: path with spaces ─────────────────────────────────────────────
    {
        cfg.tools.working_dir = "/tmp";
        Agent a(cfg);
        ::system("mkdir -p '/tmp/dir with spaces'");
        ::system("echo 'hello' > '/tmp/dir with spaces/file.txt'");
        auto r = run_task(a, "прочитай файл '/tmp/dir with spaces/file.txt' и скажи что в нём");
        check("нашёл содержимое", r.find("hello") != std::string::npos);
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cout << "\n" << DIM << "────────────────────────────\n" << RESET;
    std::cout << GRN << g_pass << " passed  " << RESET;
    std::cout << (g_fail ? RED : DIM) << g_fail << " failed" << RESET << "\n";
    return g_fail ? 1 : 0;
}
