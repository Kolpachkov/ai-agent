#include "agent.hpp"
#include "config.hpp"
#include "tools.hpp"

#include <nlohmann/json.hpp>
#include <ncurses.h>
#include <locale.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <memory>
#include <unistd.h>
#include <linux/limits.h>
#include <fcntl.h>

namespace fs = std::filesystem;

// ── Color pairs (Gruvbox dark) ────────────────────────────────────────────────
static constexpr int CP_HEADER    = 1;
static constexpr int CP_USER      = 2;
static constexpr int CP_RESP      = 3;
static constexpr int CP_TOOL      = 4;
static constexpr int CP_THINK     = 5;
static constexpr int CP_STATUS    = 6;
static constexpr int CP_ERROR     = 7;
static constexpr int CP_DIM       = 8;
static constexpr int CP_HDR_BUILD = 9;   // BUILD: dark on yellow
static constexpr int CP_HDR_PLAN  = 10;  // PLAN:  dark on teal
static constexpr int CP_HDR_LOOP  = 11;  // LOOP:  dark on green
static constexpr int CP_BG        = 12;  // output window background fill

static const char* SPINNER[]   = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
static constexpr int SPINNER_N = 10;
static constexpr int OUT_PAD   = 4;

// ── Global state ──────────────────────────────────────────────────────────────
static std::atomic<bool> g_interrupted{false};
static std::atomic<bool> g_generating{false};
static std::atomic<bool> g_resize_pending{false};
static int               g_spinner_frame{0};
static int               g_scroll_top{-1};   // -1 = auto, >= 0 = frozen line
static std::chrono::steady_clock::time_point g_gen_start;

// autonomous loop: gen_thread sets g_loop_next when response contains <next>…</next>
static std::atomic<bool>     g_loop_enabled{false};
static std::string           g_loop_next;
static std::mutex            g_loop_mu;

// ask_user: agent thread sets g_ask_pending, main thread reads the question,
// calls read_line_tui, stores answer, clears g_ask_pending, notifies CV.
static std::atomic<bool>     g_ask_pending{false};
static std::string           g_ask_question;
static std::string           g_ask_answer;
static std::mutex            g_ask_mu;
static std::condition_variable g_ask_cv;

static void handle_sigwinch(int) { g_resize_pending = true; }
static void handle_sigint(int)   { g_interrupted = true; }

// ── Thread-safe output buffer ─────────────────────────────────────────────────
static std::mutex              g_out_mu;
static std::deque<std::string> g_lines;
static bool                    g_dirty{false};

static void out_push(const std::string& text) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    if (g_lines.empty()) g_lines.push_back("");
    for (char c : text) {
        if (c == '\n') g_lines.push_back("");
        else           g_lines.back() += c;
    }
    g_dirty = true;
}

// ── UTF-8 helpers ─────────────────────────────────────────────────────────────
static std::string strip_ansi(const std::string& s) {
    std::string r;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\033') { r += s[i]; continue; }
        if (i + 1 >= s.size()) break;
        const char next = s[++i];
        if (next == '[') {
            // CSI: consume until final byte [0x40–0x7E]
            while (i + 1 < s.size()) {
                const unsigned char c = (unsigned char)s[++i];
                if (c >= 0x40 && c <= 0x7E) break;
            }
        } else if (next == ']') {
            // OSC: consume until BEL or ST
            while (i + 1 < s.size() && s[i+1] != '\007' && s[i+1] != '\033') ++i;
            if (i + 1 < s.size() && s[i+1] == '\007') ++i;
        }
        // else: two-char ESC sequence — both already consumed (ESC + next)
    }
    return r;
}

static int utf8_cols(const std::string& s) {
    int c = 0;
    for (size_t i = 0; i < s.size(); ) {
        unsigned char b = (unsigned char)s[i];
        if      (b < 0x80) { c++; i++;    }
        else if (b < 0xE0) { c++; i += 2; }
        else if (b < 0xF0) { c++; i += 3; }
        else               { c++; i += 4; }
    }
    return c;
}
// Truncate to at most max_cols display columns; append "…" if cut.
static std::string utf8_trunc(const std::string& s, int max_cols) {
    if (utf8_cols(s) <= max_cols) return s;
    int cols = 0;
    size_t i = 0;
    while (i < s.size() && cols < max_cols - 1) {
        unsigned char b = (unsigned char)s[i];
        i += (b < 0x80) ? 1 : (b < 0xE0) ? 2 : (b < 0xF0) ? 3 : 4;
        cols++;
    }
    return s.substr(0, i) + "…";
}
// Pad with spaces to exactly target_cols display columns.
static std::string utf8_pad(const std::string& s, int target_cols) {
    const int c = utf8_cols(s);
    return c >= target_cols ? s : s + std::string(target_cols - c, ' ');
}

// Bytes in s[0..len) that fit within max_cols display columns
static int bytes_for_cols(const char* s, int len, int max_cols) {
    int cols = 0, i = 0;
    while (i < len) {
        if (cols >= max_cols) return i;
        const unsigned char c = (unsigned char)s[i];
        const int step = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        if (i + step > len) return i;
        i += step; cols++;
    }
    return i;
}

// Word-aware chunk split: returns {bytes_to_display, bytes_to_advance}.
// Tries to break at the last space within cw columns; falls back to hard break.
static std::pair<int,int> word_chunk(const char* s, int len, int cw) {
    const int raw = bytes_for_cols(s, len, cw);
    if (raw >= len) return {raw, raw};          // last segment, no wrap
    // Walk back within the raw chunk to find the last ASCII space
    int sp = raw;
    while (sp > 0 && (unsigned char)s[sp-1] != ' ') --sp;
    int display = (sp > 0) ? sp - 1 : raw;
    int advance = (sp > 0) ? sp     : raw;
    // Skip all consecutive spaces so the next chunk never starts with one
    while (advance < len && s[advance] == ' ') advance++;
    return {display, advance};
}


// ── Per-line color pair ───────────────────────────────────────────────────────
static int line_cp(const std::string& line) {
    if (line.empty()) return 0;
    auto sw = [&](const char* p){ return line.find(p) == 0; };
    if (sw("◆ >"))       return CP_USER;
    if (sw("  ╰─"))      return CP_DIM;
    if (sw("  ↑"))       return CP_DIM;
    if (sw("  ⦿"))       return CP_TOOL;
    if (sw("  ✓"))       return CP_TOOL;
    if (sw("  ✗"))       return CP_ERROR;
    if (sw("  ⊕"))       return CP_TOOL;   // diff: added line (green)
    if (sw("  ⊖"))       return CP_ERROR;  // diff: removed line (red)
    if (sw("[thinking"))  return CP_THINK;
    if (sw("[error"))     return CP_ERROR;
    if (sw("[inter") || sw("[busy")) return CP_DIM;
    if (sw("[mode")  || sw("[history") || sw("[model")
     || sw("[verbose")|| sw("[loading")|| sw("[session")
     || sw("[current")|| sw("[mouse")  || sw("  ↑ session")) return CP_STATUS;
    return CP_RESP;
}

// ── Gruvbox color init ────────────────────────────────────────────────────────
static void init_colors() {
    if (!has_colors()) return;
    if (COLORS >= 256) {
        init_pair(CP_HEADER,    235, 214);
        init_pair(CP_USER,      214,  -1);
        init_pair(CP_RESP,      223,  -1);
        init_pair(CP_TOOL,      108,  -1);
        init_pair(CP_THINK,     175,  -1);
        init_pair(CP_STATUS,    142,  -1);
        init_pair(CP_ERROR,     167,  -1);
        init_pair(CP_DIM,       245,  -1);
        init_pair(CP_HDR_BUILD, 235, 214);  // dark on Gruvbox yellow
        init_pair(CP_HDR_PLAN,  235,  66);  // dark on Gruvbox aqua
        init_pair(CP_HDR_LOOP,  235, 142);  // dark on Gruvbox green
        init_pair(CP_BG,        223, 235);  // cream on Gruvbox dark bg
    } else {
        init_pair(CP_HEADER,    COLOR_BLACK,  COLOR_YELLOW);
        init_pair(CP_USER,      COLOR_YELLOW, -1);
        init_pair(CP_RESP,      -1,           -1);
        init_pair(CP_TOOL,      COLOR_GREEN,  -1);
        init_pair(CP_THINK,     COLOR_CYAN,   -1);
        init_pair(CP_STATUS,    COLOR_GREEN,  -1);
        init_pair(CP_ERROR,     COLOR_RED,    -1);
        init_pair(CP_DIM,       COLOR_WHITE,  -1);
        init_pair(CP_HDR_BUILD, COLOR_BLACK,  COLOR_YELLOW);
        init_pair(CP_HDR_PLAN,  COLOR_BLACK,  COLOR_CYAN);
        init_pair(CP_HDR_LOOP,  COLOR_BLACK,  COLOR_GREEN);
        init_pair(CP_BG,        -1,           COLOR_BLACK);
    }
}

// ── Strip markdown ────────────────────────────────────────────────────────────
static std::string strip_md(const std::string& line) {
    std::string r; r.reserve(line.size());
    for (size_t i = 0; i < line.size(); ) {
        const unsigned char c = (unsigned char)line[i];
        if (c == '`' && i+2 < line.size() && line[i+1]=='`' && line[i+2]=='`') { i+=3; continue; }
        if (c == '*' && i+1 < line.size() && line[i+1]=='*') {
            i += (i+2 < line.size() && line[i+2]=='*') ? 3 : 2; continue;
        }
        if (c == '*' || c == '_') {
            const bool lw = i>0 && (std::isalnum((unsigned char)line[i-1]) || (unsigned char)line[i-1]>127);
            const bool rw = i+1<line.size() && (std::isalnum((unsigned char)line[i+1]) || (unsigned char)line[i+1]>127);
            if (!lw || !rw) { i++; continue; }
        }
        if (c == '`') { i++; continue; }
        const size_t step = (c<0x80)?1:(c<0xE0)?2:(c<0xF0)?3:4;
        for (size_t j=0; j<step && i<line.size(); ++j) r += line[i++];
    }
    size_t h = 0;
    while (h < r.size() && r[h]=='#') h++;
    if (h>0 && h<r.size() && r[h]==' ') r = r.substr(h+1);
    if (!r.empty() && r[0]=='>' && (r.size()<2 || r[1]==' '))
        r = r.substr(r.size()>1 ? 2 : 1);
    return r;
}

// ── ncurses windows ───────────────────────────────────────────────────────────
static WINDOW* g_hdr = nullptr;
static WINDOW* g_out = nullptr;
static WINDOW* g_inp = nullptr;
static int     g_out_h = 0;

static void create_windows() {
    g_out_h = LINES - 2;
    if (g_out_h < 1) g_out_h = 1;
    g_hdr = newwin(1, COLS, 0, 0);
    g_out = newwin(g_out_h, COLS, 1, 0);
    if (has_colors()) wbkgd(g_out, COLOR_PAIR(CP_BG)); // opaque bg, no wallpaper bleed
    g_inp = newwin(1, COLS, LINES-1, 0);
    wtimeout(g_inp, 80);
    keypad(g_inp, TRUE);
}
static void destroy_windows() {
    if (g_hdr) { delwin(g_hdr); g_hdr=nullptr; }
    if (g_out) { delwin(g_out); g_out=nullptr; }
    if (g_inp) { delwin(g_inp); g_inp=nullptr; }
}
static void resize_windows() {
    destroy_windows(); resizeterm(0,0); endwin(); refresh();
    create_windows(); g_dirty = true;
}

static int hdr_pair(AgentMode mode) {
    if (g_loop_enabled)               return CP_HDR_LOOP;
    if (mode == AgentMode::Plan)      return CP_HDR_PLAN;
    return CP_HDR_BUILD;
}

static void draw_header(AgentMode mode, const std::string& model_name) {
    if (!g_hdr) return;
    werase(g_hdr);
    const attr_t ha = (has_colors() ? COLOR_PAIR(hdr_pair(mode)) : A_REVERSE) | A_BOLD;
    wbkgd(g_hdr, ha); wattron(g_hdr, ha);
    const char* ms = (mode == AgentMode::Plan) ? "PLAN" : "BUILD";
    std::string spin;
    if (g_generating) {
        const int secs = (int)std::chrono::duration<double>(
            std::chrono::steady_clock::now() - g_gen_start).count();
        spin = std::string(" ") + SPINNER[g_spinner_frame%SPINNER_N];
        if (secs >= 3) { spin += " "; spin += std::to_string(secs); spin += "с"; }
    }
    std::string stag = (g_scroll_top >= 0) ? " [↑]" : "";
    std::string loop_tag = g_loop_enabled ? " ⟳" : "";
    std::string left = " ◆ ai-agent  |  " + std::string(ms) + loop_tag + spin + stag + "  |  " + model_name;
    const std::string right = " колесо: скрол  Shift+drag: копи  ESC: стоп ";
    const int pad = COLS - utf8_cols(left) - utf8_cols(right);
    for (int i=0; i<pad; ++i) left += ' ';
    left += right;
    mvwaddstr(g_hdr, 0, 0, left.c_str());
    wattroff(g_hdr, ha); wrefresh(g_hdr);
}

// ── flush_output: pre-expand to visual rows, slice last g_out_h ──────────────
static void flush_output() {
    if (!g_out) return;
    std::lock_guard<std::mutex> lk(g_out_mu);
    if (!g_dirty) return;

    const int total = (int)g_lines.size();
    const int cw = std::max(1, COLS - 2*OUT_PAD);

    struct VRow { std::string text; int cp; };
    std::vector<VRow> vrows;
    vrows.reserve(std::max(total, 1) * 2);
    std::vector<int> lstart; // logical line i → first VRow index
    lstart.reserve(total);

    for (int i = 0; i < total; ++i) {
        lstart.push_back((int)vrows.size());
        const auto& raw = g_lines[i];
        const int cp = has_colors() ? line_cp(raw) : 0;
        const std::string line = (cp == CP_RESP) ? strip_md(raw) : raw;
        if (line.empty()) { vrows.push_back({"", cp}); continue; }
        int bp = 0, len = (int)line.size();
        while (bp < len) {
            const auto [disp, adv] = word_chunk(line.c_str()+bp, len-bp, cw);
            if (adv == 0) break;
            vrows.push_back({line.substr(bp, disp), cp});
            bp += adv;
        }
    }

    const int vtotal = (int)vrows.size();
    int vstart;
    if (g_scroll_top >= 0 && total > 0) {
        vstart = lstart[std::min(g_scroll_top, total-1)];
    } else {
        vstart = std::max(0, vtotal - g_out_h);
    }

    werase(g_out);
    for (int row = 0; row < g_out_h; ++row) {
        const int vi = vstart + row;
        if (vi >= vtotal) break;
        const auto& vr = vrows[vi];
        if (vr.text.empty()) continue;
        if (vr.cp) wattron(g_out, COLOR_PAIR(vr.cp));
        mvwaddstr(g_out, row, OUT_PAD, vr.text.c_str());
        if (vr.cp) wattroff(g_out, COLOR_PAIR(vr.cp));
    }
    g_dirty = false;
    wrefresh(g_out);
}

// ── Scroll ────────────────────────────────────────────────────────────────────
static void output_scroll_up(int n) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    const int total = (int)g_lines.size();
    const int cur   = (g_scroll_top < 0) ? std::max(0, total-g_out_h) : g_scroll_top;
    g_scroll_top = std::max(0, cur-n);
    g_dirty = true;
}
static void output_scroll_down(int n) {
    std::lock_guard<std::mutex> lk(g_out_mu);
    const int total = (int)g_lines.size();
    if (g_scroll_top < 0) return;
    g_scroll_top += n;
    if (g_scroll_top + g_out_h >= total) g_scroll_top = -1;
    g_dirty = true;
}

// ── Input state ───────────────────────────────────────────────────────────────
struct InputState {
    std::string buf;
    int cursor=0, view_col=0;
    std::vector<std::string> history;
    int hist_idx=-1;
    std::string hist_saved;

    static int prev_utf8(const std::string& s, int p) {
        if (p<=0) return 0;
        do { --p; } while (p>0 && ((unsigned char)s[p]&0xC0)==0x80);
        return p;
    }
    static int next_utf8(const std::string& s, int p) {
        if (p>=(int)s.size()) return (int)s.size();
        do { ++p; } while (p<(int)s.size() && ((unsigned char)s[p]&0xC0)==0x80);
        return p;
    }
    static int cols_to(const std::string& s, int bp) {
        int col=0;
        for (int i=0; i<bp && i<(int)s.size(); ) {
            unsigned char c=(unsigned char)s[i];
            if      (c<0x80){col++;i++;}
            else if (c<0xE0){col++;i+=2;}
            else if (c<0xF0){col++;i+=3;}
            else            {col++;i+=4;}
        }
        return col;
    }
    void insert(unsigned char c){ buf.insert(buf.begin()+cursor,(char)c); ++cursor; }
    void backspace(){ if(cursor<=0) return; int p=prev_utf8(buf,cursor); buf.erase(p,cursor-p); cursor=p; }
    void del_fwd(){ if(cursor>=(int)buf.size()) return; buf.erase(cursor,next_utf8(buf,cursor)-cursor); }
    void left() { cursor=prev_utf8(buf,cursor); }
    void right(){ cursor=next_utf8(buf,cursor); }
    void home() { cursor=0; }
    void end()  { cursor=(int)buf.size(); }
    void hist_up(){
        if(history.empty()) return;
        if(hist_idx<0){hist_saved=buf;hist_idx=(int)history.size()-1;}
        else if(hist_idx>0) --hist_idx;
        buf=history[hist_idx]; cursor=(int)buf.size();
    }
    void hist_down(){
        if(hist_idx<0) return;
        if(hist_idx<(int)history.size()-1){++hist_idx;buf=history[hist_idx];}
        else{hist_idx=-1;buf=hist_saved;}
        cursor=(int)buf.size();
    }
    std::string submit(){
        std::string r=buf;
        if(!buf.empty()&&(history.empty()||history.back()!=buf)) history.push_back(buf);
        buf.clear(); cursor=0; hist_idx=-1;
        return r;
    }
    void clear(){ buf.clear(); cursor=0; }
};

static constexpr int PFX_COLS = 4; // "◆ > "

static void draw_input(InputState& inp, AgentMode mode) {
    if (!g_inp) return;
    const attr_t ba = has_colors() ? (COLOR_PAIR(hdr_pair(mode)) | A_BOLD) : A_BOLD;
    wbkgd(g_inp, ba);   // must be before werase so erase fills with mode color
    werase(g_inp);
    wattron(g_inp, ba);
    const int avail = COLS - PFX_COLS;
    if (avail <= 0) { wattroff(g_inp, ba); wrefresh(g_inp); return; }
    const int cur_col = inp.cols_to(inp.buf, inp.cursor);
    if (cur_col < inp.view_col) inp.view_col = cur_col;
    else if (cur_col >= inp.view_col+avail) inp.view_col = cur_col-avail+1;
    int vbyte = 0;
    for (int vc=0; vc<inp.view_col && vbyte<(int)inp.buf.size(); ++vc)
        vbyte = inp.next_utf8(inp.buf, vbyte);
    mvwaddstr(g_inp, 0, 0, "◆ > ");
    int ebyte=vbyte, cw=0;
    while (ebyte<(int)inp.buf.size() && cw<avail){ ebyte=inp.next_utf8(inp.buf,ebyte); ++cw; }
    if (vbyte<ebyte) waddnstr(g_inp, inp.buf.c_str()+vbyte, ebyte-vbyte);
    wattroff(g_inp, ba);
    wmove(g_inp, 0, PFX_COLS+cur_col-inp.view_col);
    wrefresh(g_inp);
}

// ── read_line_tui: blocking input inside ncurses (no endwin) ─────────────────
static std::string read_line_tui(const std::string& prompt) {
    std::string line;
    g_dirty = true;
    while (true) {
        flush_output();
        if (g_inp) {
            werase(g_inp);
            const attr_t pa = has_colors() ? (COLOR_PAIR(CP_STATUS)|A_BOLD) : A_BOLD;
            wattron(g_inp, pa);
            mvwaddstr(g_inp, 0, 0, prompt.c_str());
            wattroff(g_inp, pa);
            waddstr(g_inp, line.c_str());
            wmove(g_inp, 0, utf8_cols(prompt)+(int)utf8_cols(line));
            wrefresh(g_inp);
        }
        const int ch = wgetch(g_inp);
        if (ch == ERR) continue;
        if (ch == '\n' || ch == '\r' || ch == KEY_ENTER) return line;
        if (ch == 27) return "";
        if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (!line.empty()) {
                do { line.pop_back(); }
                while (!line.empty() && ((unsigned char)line.back()&0xC0)==0x80);
            }
        } else if (ch >= 32 && ch <= 255) {
            line += (char)(unsigned char)ch;
        }
    }
}

// ── Tab completion ────────────────────────────────────────────────────────────
static const std::vector<std::string> CMDS = {
    "/plan","/build","/model","/reset","/sessions","/delete","/verbose","/mouse","/exit"
};
static void tab_complete(InputState& inp) {
    if (inp.buf.empty() || inp.buf[0]!='/') return;
    std::vector<std::string> m;
    for (const auto& c : CMDS)
        if (c.size()>=inp.buf.size() && c.substr(0,inp.buf.size())==inp.buf) m.push_back(c);
    if (m.size()==1){ inp.buf=m[0]; inp.cursor=(int)inp.buf.size(); }
    else if (m.size()>1){ std::string s="  "; for(auto&c:m) s+=c+"  "; out_push(s+"\n"); }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
static std::string expand_home(const std::string& path) {
    if (path.empty() || path[0]!='~') return path;
    const char* h = getenv("HOME");
    return h ? std::string(h)+path.substr(1) : path;
}
static std::string make_session_slug(const std::string& msg) {
    std::string slug; bool last_dash=false;
    for (unsigned char c : msg) {
        if (c>=0x80||std::isalnum(c)){ slug+=(char)c; last_dash=false; }
        else if (std::isspace(c)||c=='-'||c=='_'){
            if(!last_dash&&!slug.empty()){ slug+='-'; last_dash=true; }
        }
        if (slug.size()>=40) break;
    }
    while (!slug.empty()&&slug.back()=='-') slug.pop_back();
    return slug.empty()?"session":slug;
}
static std::string unique_session_path(const std::string& dir, const std::string& slug) {
    const std::string base = dir+"/"+slug;
    std::string path = base+".json";
    for (int i=2; fs::exists(path); ++i) path = base+"_"+std::to_string(i)+".json";
    return path;
}

// ── Session management ────────────────────────────────────────────────────────
struct SessionEntry { std::string path,title; int count=0; fs::file_time_type mtime; };

static std::vector<SessionEntry> scan_sessions(const std::string& dir_raw) {
    const std::string dir = expand_home(dir_raw);
    std::vector<SessionEntry> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir,ec)) {
        if (ec){ ec.clear(); continue; }
        if (e.path().extension()!=".json") continue;
        std::ifstream f(e.path());
        if (!f.is_open()) continue;
        nlohmann::json j;
        try { j=nlohmann::json::parse(f); } catch(...){ continue; }
        if (!j.is_array()||j.empty()) continue;
        SessionEntry se; se.path=e.path().string(); se.count=(int)j.size();
        std::error_code mec; se.mtime=e.last_write_time(mec);
        for (const auto& msg:j) {
            if (msg.value("role","")=="user") {
                std::string t=msg.value("content","");
                while (!t.empty()&&(t.back()=='\n'||t.back()=='\r')) t.pop_back();
                se.title = utf8_trunc(t, 42); break;
            }
        }
        if (se.title.empty()) se.title=e.path().stem().string();
        out.push_back(std::move(se));
    }
    std::sort(out.begin(),out.end(),[](const SessionEntry&a,const SessionEntry&b){ return a.mtime>b.mtime; });
    return out;
}

// ── Display loaded session history in output window ───────────────────────────
static void display_session_history(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    nlohmann::json j;
    try { j = nlohmann::json::parse(f); } catch(...) { return; }
    if (!j.is_array() || j.empty()) return;

    out_push("\n  ─── история сессии ───\n\n");
    for (const auto& msg : j) {
        const std::string role    = msg.value("role","");
        const std::string content = msg.value("content","");
        if (role == "system") continue;
        if (role == "user") {
            // Show first line of user message only
            const auto nl = content.find('\n');
            const std::string first = nl == std::string::npos ? content : content.substr(0,nl);
            out_push("◆ > " + (first.size()>80 ? first.substr(0,77)+"…" : first) + "\n");
        } else if (role == "assistant") {
            // Show first 4 lines of assistant reply
            std::istringstream ss(content);
            std::string line; int shown=0;
            while (std::getline(ss,line) && shown<4) {
                if (!line.empty()) { out_push("  " + line + "\n"); ++shown; }
            }
            if (ss.peek() != EOF) out_push("  …\n");
        }
        out_push("\n");
    }
    out_push("  ─── конец истории ───\n\n");
}

// ── Session selection (runs inside TUI) ───────────────────────────────────────
static int tui_pick_session(const std::vector<SessionEntry>& sessions) {
    std::string s = "\n  ╔══ Сессии ══════════════════════════════════════════════╗\n";
    s += "  ║  [0]  " + utf8_pad("новая сессия", 49) + "║\n";
    for (int i=0; i<(int)sessions.size(); ++i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "  (%2d) ║", sessions[i].count);
        s += "  ║  [" + std::to_string(i+1) + "]  "
           + utf8_pad(sessions[i].title, 42) + buf + "\n";
    }
    s += "  ╚════════════════════════════════════════════════════════╝\n";
    out_push(s);
    const std::string ans = read_line_tui("  номер или Enter для новой: ");
    out_push("\n");
    if (ans.empty()) return 0;
    try {
        const int n = std::stoi(ans);
        if (n>=0 && n<=(int)sessions.size()) return n;
    } catch (...) {}
    return 0;
}

// ── /sessions command ─────────────────────────────────────────────────────────
static void cmd_sessions(const std::string& sessions_dir, const std::string& cur_path) {
    if (sessions_dir.empty()) { out_push("[sessions_dir not set]\n"); return; }
    const auto list = scan_sessions(sessions_dir);
    if (list.empty()) { out_push("[no saved sessions]\n"); return; }
    std::string s = "\n  Sessions:\n";
    for (int i=0; i<(int)list.size(); ++i) {
        char buf[300];
        snprintf(buf,sizeof(buf),"  [%d]  %-50s  (%d msg)%s",
                 i+1, list[i].title.c_str(), list[i].count,
                 list[i].path==cur_path?"  ← current":"");
        s += std::string(buf)+"\n";
    }
    out_push(s+"\n");
}

// ── /delete command ───────────────────────────────────────────────────────────
static bool cmd_delete(const std::string& sessions_dir, const std::string& cur_path) {
    if (sessions_dir.empty()) { out_push("[sessions_dir not set]\n"); return false; }
    const auto list = scan_sessions(sessions_dir);
    if (list.empty()) { out_push("[no saved sessions]\n"); return false; }
    std::string menu = "\n  Delete session:\n";
    for (int i=0; i<(int)list.size(); ++i) {
        char buf[300];
        snprintf(buf,sizeof(buf),"  [%d]  %-50s  (%d msg)%s",
                 i+1, list[i].title.c_str(), list[i].count,
                 list[i].path==cur_path?"  ← current":"");
        menu += std::string(buf)+"\n";
    }
    out_push(menu);
    const std::string line = read_line_tui("  numbers (comma-sep) or Enter to cancel: ");
    if (line.empty()) { out_push("[cancelled]\n"); return false; }
    bool deleted_cur = false;
    std::istringstream ss(line); std::string tok;
    while (std::getline(ss,tok,',')) {
        const size_t a=tok.find_first_not_of(" \t"), b=tok.find_last_not_of(" \t");
        if (a==std::string::npos) continue;
        tok=tok.substr(a,b-a+1);
        try {
            const int n=std::stoi(tok);
            if (n<1||n>(int)list.size()){ out_push("  [invalid: "+tok+"]\n"); continue; }
            std::error_code ec; fs::remove(list[n-1].path,ec);
            out_push(ec?"  ✗ "+list[n-1].title+"\n":"  ✓ deleted: "+list[n-1].title+"\n");
            if (!ec && list[n-1].path==cur_path) deleted_cur=true;
        } catch(...){ out_push("  [not a number: "+tok+"]\n"); }
    }
    return deleted_cur;
}

// ── /model command ────────────────────────────────────────────────────────────
static std::string model_short(const std::string& path) {
    std::string s = fs::path(path).stem().string();
    if (s.size()>20) s=s.substr(0,17)+"...";
    return s;
}
static std::vector<std::string> scan_models(const std::string& dir="./models") {
    std::vector<std::string> out; std::error_code ec;
    for (const auto& e:fs::directory_iterator(dir,ec)) {
        if (ec){ ec.clear(); continue; }
        if (e.path().extension()==".gguf") out.push_back(e.path().string());
    }
    std::sort(out.begin(),out.end()); return out;
}
static std::string cmd_model(const std::string& current_model) {
    const auto models = scan_models();
    if (models.empty()) { out_push("[no .gguf files in ./models/]\n"); return ""; }
    std::string menu = "\n  Models:\n  [0]  keep current ("+model_short(current_model)+")\n";
    for (int i=0; i<(int)models.size(); ++i)
        menu += "  ["+std::to_string(i+1)+"]  "+model_short(models[i])
                +(models[i]==current_model?"  ← current":"")+"\n";
    out_push(menu);
    const std::string line = read_line_tui("  choose [0-"+std::to_string(models.size())+"]: ");
    if (line.empty()) return "";
    try {
        const int n=std::stoi(line);
        if (n==0) return "";
        if (n>=1&&n<=(int)models.size()) return models[n-1];
    } catch(...) {}
    out_push("[invalid choice]\n");
    return "";
}

// Write escape sequences directly to the terminal device, bypassing ncurses
// and stdout buffering (ncurses may use a different internal stream).
static void tty_write(const char* seq) {
    int fd = open("/dev/tty", O_WRONLY | O_NOCTTY);
    if (fd < 0) return;
    write(fd, seq, strlen(seq));
    close(fd);
}

// ── Mouse toggle ──────────────────────────────────────────────────────────────
// Mouse is ON by default (scroll wheel works). /mouse turns it OFF so the
// terminal allows native drag-select; use Shift+drag when mouse is ON.
static bool g_mouse_on = true;
static void toggle_mouse() {
    g_mouse_on = !g_mouse_on;
    if (g_mouse_on) {
        mousemask(ALL_MOUSE_EVENTS|REPORT_MOUSE_POSITION, nullptr);
        tty_write("\033[?1007l");  // disable alternate scroll → wheel gives BUTTON4/5 events
        out_push("[mouse: on — колёсо прокручивает, Shift+drag для копирования]\n");
    } else {
        mousemask(0, nullptr);
        tty_write("\033[?1007h");  // restore alternate scroll
        out_push("[mouse: off — выделяй и копируй мышью, /mouse чтобы вернуть колесо]\n");
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
// Return the directory of the running executable via /proc/self/exe.
static fs::path exe_dir() {
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return fs::current_path();
    buf[n] = '\0';
    return fs::path(buf).parent_path();
}

int main(int argc, char** argv) {
    // Default config: look for config.json next to the binary, then one level
    // up (project root when binary lives in build/), then fall back to CWD.
    std::string config_path;
    {
        const fs::path d = exe_dir();
        // Prefer config.json in the project root (parent of build/) over any
        // stale copy that might sit next to the binary in build/.
        if      (fs::exists(d.parent_path() / "config.json")) config_path = (d.parent_path() / "config.json").string();
        else if (fs::exists(d / "config.json"))               config_path = (d / "config.json").string();
        else                                                   config_path = "./config.json";
    }

    bool override_verbose=false, mode_override=false;
    AgentMode override_mode = AgentMode::Build;

    for (int i=1; i<argc; ++i) {
        const std::string arg(argv[i]);
        if      (arg=="--config" && i+1<argc) config_path=argv[++i];
        else if (arg=="--mode"   && i+1<argc){ override_mode=mode_from_string(argv[++i]); mode_override=true; }
        else if (arg=="--verbose") override_verbose=true;
        else if (arg=="--help"||arg=="-h"){
            std::cout<<"Usage: ai-agent [--config PATH] [--mode plan|build] [--verbose]\n"; return 0;
        }
    }

    AppConfig cfg;
    try { cfg=load_config(config_path); }
    catch(const std::exception& e){ std::cerr<<"[error] "<<e.what()<<"\n"; return 1; }

    // Resolve model path relative to config file's directory so relative
    // paths in config.json work regardless of the working directory.
    if (!cfg.model.path.empty() && !fs::path(cfg.model.path).is_absolute()) {
        std::error_code ec;
        const fs::path config_dir = fs::canonical(config_path, ec).parent_path();
        if (!ec) cfg.model.path = (config_dir / cfg.model.path).lexically_normal().string();
    }
    if (override_verbose) cfg.agent.verbose=true;
    if (mode_override)    cfg.agent.mode=override_mode;
    // Always operate in the directory where the binary was invoked
    cfg.tools.working_dir = fs::current_path().string();

    // Create sessions dir early (silent)
    const std::string sessions_dir_exp = expand_home(cfg.agent.sessions_dir);
    if (!cfg.agent.sessions_dir.empty()) {
        std::error_code ec; fs::create_directories(sessions_dir_exp,ec);
    }
    if (cfg.tools.enabled) {
        std::error_code ec; fs::create_directories(cfg.tools.working_dir,ec);
    }

    // Load model before ncurses (llama.cpp prints to stderr, let it show)
    std::cerr<<"[info] Loading model: "<<cfg.model.path<<" ...\n";
    std::unique_ptr<Agent> agent;
    try { agent = std::make_unique<Agent>(cfg); }
    catch(const std::exception& e){ std::cerr<<"[error] "<<e.what()<<"\n"; return 1; }
    set_stop_flag(&g_interrupted);
    set_ask_user_fn([](const std::string& question) -> std::string {
        { std::lock_guard<std::mutex> lk(g_ask_mu); g_ask_question = question; g_ask_answer.clear(); }
        g_ask_pending.store(true);
        std::unique_lock<std::mutex> lk(g_ask_mu);
        g_ask_cv.wait(lk, []{ return !g_ask_pending.load(); });
        return g_ask_answer;
    });
    std::cerr<<"[info] Ready.\n";

    // ── ncurses ───────────────────────────────────────────────────────────────
    setlocale(LC_ALL,"");
    setenv("ESCDELAY","25",1);
    initscr(); cbreak(); noecho(); curs_set(1);
    if (has_colors()){ start_color(); use_default_colors(); init_colors(); }
    mousemask(ALL_MOUSE_EVENTS|REPORT_MOUSE_POSITION, nullptr);
    mouseinterval(0);
    std::signal(SIGWINCH, handle_sigwinch);
    create_windows();

    std::string model_name = model_short(cfg.model.path);
    InputState  inp;
    std::thread gen_thread;
    bool        session_named = false;

    draw_header(agent->mode(), model_name);
    flush_output();
    draw_input(inp, agent->mode());

    std::signal(SIGINT, handle_sigint);

    // Always start a fresh session (no selection menu)
    out_push("\n  ◆ " + fs::current_path().string() + "\n\n");

    // ── Gen-thread factory ────────────────────────────────────────────────────
    // Defined here so it can be used both by the Enter handler and auto-loop.
    auto make_gen_thread = [&](const std::string& input_line) -> std::thread {
        return std::thread([&, input_line]() {
            const auto t0 = std::chrono::steady_clock::now();
            g_gen_start = t0;
            std::string full_resp;
            try {
                auto stream_cb = [&](const std::string& piece) -> bool {
                    full_resp += piece;
                    out_push(strip_ansi(piece));
                    return !g_interrupted;
                };
                auto think_cb = [&]() { ++g_spinner_frame; };
                agent->chat(input_line, stream_cb, think_cb);
                out_push("\n");
                agent->save_history();

                // Extract <next>task</next> for autonomous loop (only when enabled)
                if (!g_interrupted && g_loop_enabled) {
                    const size_t ns = full_resp.find("<next>");
                    const size_t ne = full_resp.find("</next>");
                    if (ns != std::string::npos && ne != std::string::npos && ne > ns) {
                        std::string task = full_resp.substr(ns + 6, ne - ns - 6);
                        const size_t s = task.find_first_not_of(" \n\r\t");
                        const size_t e = task.find_last_not_of(" \n\r\t");
                        if (s != std::string::npos) {
                            task = task.substr(s, e - s + 1);
                            out_push("[⟳ авто: " + task + "]\n");
                            std::lock_guard<std::mutex> lk(g_loop_mu);
                            g_loop_next = task;
                        }
                    }
                }
            } catch(const std::exception& e) {
                out_push("\n[error: "+std::string(e.what())+"]\n");
            }
            if (g_interrupted) { out_push("[interrupted]\n"); std::lock_guard<std::mutex> lk(g_loop_mu); g_loop_next.clear(); }
            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            char tbuf[48];
            const int m = (int)(secs / 60);
            if (m == 0) snprintf(tbuf, sizeof(tbuf), "  ╰─ %.1f с\n", secs);
            else        snprintf(tbuf, sizeof(tbuf), "  ╰─ %d м %d с\n", m, (int)secs % 60);
            out_push(tbuf);
            g_generating=false; g_interrupted=false;
        });
    };

    // ── Main event loop ───────────────────────────────────────────────────────
    while (true) {
        if (g_resize_pending.exchange(false)) resize_windows();

        // auto-loop: fire next task if agent left a <next> continuation
        if (!g_generating && !g_interrupted) {
            std::string next;
            { std::lock_guard<std::mutex> lk(g_loop_mu); next = std::move(g_loop_next); }
            if (!next.empty()) {
                { std::lock_guard<std::mutex> lk(g_out_mu); g_scroll_top=-1; g_dirty=true; }
                out_push("\n◆ [⟳ авто] " + next + "\n  ╰─ \n");
                g_generating=true; g_interrupted=false;
                if (gen_thread.joinable()) gen_thread.join();
                gen_thread = make_gen_thread(next);
                continue;
            }
        }

        // ask_user: agent thread is waiting — show question and read answer
        if (g_ask_pending.load()) {
            std::string q;
            { std::lock_guard<std::mutex> lk(g_ask_mu); q = g_ask_question; }
            out_push("\n◆ [вопрос агента] " + q + "\n");
            flush_output(); draw_header(agent->mode(), model_name); draw_input(inp, agent->mode());
            const std::string ans = read_line_tui("  ответ: ");
            out_push("  ответ: " + ans + "\n\n");
            { std::lock_guard<std::mutex> lk(g_ask_mu); g_ask_answer = ans; }
            g_ask_pending.store(false);
            g_ask_cv.notify_one();
        }

        flush_output();
        draw_header(agent->mode(), model_name);
        draw_input(inp, agent->mode());

        const int ch = wgetch(g_inp);

        if (ch == ERR) {
            if (g_generating) ++g_spinner_frame;
            continue;
        }

        // ESC
        if (ch == 27) { if(g_generating) g_interrupted=true; inp.clear(); continue; }
        // Ctrl+D
        if (ch == 4) break;

        // Mouse scroll
        if (ch == KEY_MOUSE) {
            MEVENT ev;
            if (getmouse(&ev)==OK) {
                if      (ev.bstate & BUTTON4_PRESSED)  output_scroll_up(3);
#ifdef BUTTON5_PRESSED
                else if (ev.bstate & BUTTON5_PRESSED)  output_scroll_down(3);
#endif
            }
            continue;
        }

        // Input navigation
        if (ch==KEY_LEFT)  { inp.left();      continue; }
        if (ch==KEY_RIGHT) { inp.right();     continue; }
        if (ch==KEY_UP)    { inp.hist_up();   continue; }
        if (ch==KEY_DOWN)  { inp.hist_down(); continue; }
        if (ch==KEY_HOME||ch==1) { inp.home(); continue; }
        if (ch==KEY_END ||ch==5) { inp.end();  continue; }

        // Output scroll
        if (ch==KEY_PPAGE) { output_scroll_up(g_out_h/2);   continue; }
        if (ch==KEY_NPAGE) { output_scroll_down(g_out_h/2); continue; }

        // Backspace / Delete
        if (ch==KEY_BACKSPACE||ch==127||ch==8) { inp.backspace(); continue; }
        if (ch==KEY_DC) { inp.del_fwd(); continue; }

        // Tab
        if (ch=='\t') { tab_complete(inp); continue; }

        // Ctrl+K / Ctrl+U
        if (ch==11){ inp.buf.erase(inp.cursor); continue; }
        if (ch==21){ inp.buf.erase(0,inp.cursor); inp.cursor=0; continue; }

        // Enter
        if (ch=='\n'||ch=='\r'||ch==KEY_ENTER) {
            std::string line = inp.submit();
            if (line.empty()) continue;
            if (g_generating) { out_push("[busy — ESC to interrupt]\n"); continue; }

            // Scroll back to bottom on any input
            { std::lock_guard<std::mutex> lk(g_out_mu); g_scroll_top=-1; g_dirty=true; }

            if (line=="/exit"||line=="/quit") break;
            if (line=="/plan"){  agent->set_mode(AgentMode::Plan);  out_push("[mode: PLAN]\n");  continue; }
            if (line=="/build"){ agent->set_mode(AgentMode::Build); out_push("[mode: BUILD]\n"); continue; }
            if (line=="/reset"){
                agent->reset_history(); agent->set_session_path(""); session_named=false;
                out_push("[history cleared]\n"); continue;
            }
            if (line=="/sessions"){ cmd_sessions(cfg.agent.sessions_dir, agent->current_session_path()); continue; }
            if (line=="/delete"){
                if (cmd_delete(cfg.agent.sessions_dir, agent->current_session_path())) {
                    agent->reset_history(); agent->set_session_path(""); session_named=false;
                    out_push("[current session deleted — history cleared]\n");
                }
                continue;
            }
            if (line=="/model"){
                const std::string nm = cmd_model(cfg.model.path);
                if (!nm.empty()&&nm!=cfg.model.path) {
                    out_push("[loading "+model_short(nm)+" ...]\n"); flush_output();
                    try {
                        cfg.model.path=nm;
                        agent=std::make_unique<Agent>(cfg);
                        model_name=model_short(nm);
                        agent->reset_history(); session_named=false;
                        out_push("[model: "+model_name+"]\n");
                    } catch(const std::exception& e){
                        out_push("[error loading model: "+std::string(e.what())+"]\n");
                    }
                }
                continue;
            }
            if (line=="/loop"){
                g_loop_enabled = !g_loop_enabled;
                { std::lock_guard<std::mutex> lk(g_loop_mu); g_loop_next.clear(); }
                agent->set_loop_enabled(g_loop_enabled);
                out_push(g_loop_enabled ? "[loop: ON — агент будет продолжать автономно]\n"
                                        : "[loop: OFF]\n");
                continue;
            }
            if (line=="/mouse"){ toggle_mouse(); continue; }
            if (line=="/verbose"){ cfg.agent.verbose=!cfg.agent.verbose;
                out_push(cfg.agent.verbose?"[verbose: on]\n":"[verbose: off]\n"); continue; }

            // Assign session path on first real message
            if (!session_named && !cfg.agent.sessions_dir.empty()) {
                agent->set_session_path(unique_session_path(sessions_dir_exp, make_session_slug(line)));
                session_named=true;
            }

            out_push("\n◆ > "+line+"\n  ╰─ \n");

            g_generating=true; g_interrupted=false;
            if (gen_thread.joinable()) gen_thread.join();
            gen_thread = make_gen_thread(line);
            continue;
        }

        // Regular UTF-8 char
        if (ch>=32 && ch<=255) inp.insert((unsigned char)ch);
    }

    g_interrupted=true;
    if (gen_thread.joinable()) gen_thread.join();
    destroy_windows(); endwin();
    return 0;
}
