#pragma once
#include <string>
#include <vector>
#include <functional>
#include <nlohmann/json.hpp>

struct ToolResult {
    bool        success = true;
    std::string output;
};

struct Tool {
    std::string name;
    std::string description;
    std::string params_doc;
    std::function<ToolResult(const nlohmann::json&)> fn;
};

class ToolRegistry {
public:
    void        set_working_dir(const std::string& dir);
    void        add(Tool tool);
    bool        has(const std::string& name) const;
    ToolResult  call(const std::string& name, const nlohmann::json& args) const;
    std::string system_section() const;  // Injected into system prompt

private:
    std::vector<Tool> tools_;
    std::string       working_dir_ = ".";
};

ToolRegistry make_default_tools(const std::string& working_dir = ".");
ToolRegistry make_readonly_tools(const std::string& working_dir = ".");  // list_dir, read_file, search_files only
