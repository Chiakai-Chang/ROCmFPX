// Replays a corpus of raw model outputs (tests/tool-corpus/cases.json) through the PEG chat parser built from a
// chat template, and reports which tool calls were extracted. Usage: tool-corpus <template.jinja> <cases.json> [-v]
#include "chat.h"
#include "common.h"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + p); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 3) { std::cerr << "usage: tool-corpus <template> <cases.json> [-v]\n"; return 2; }
    bool verbose = argc > 3 && std::string(argv[3]) == "-v";
    common_chat_templates_ptr tmpls(common_chat_templates_init(nullptr, slurp(argv[1])));
    auto cases = nlohmann::ordered_json::parse(slurp(argv[2]));
    int pass = 0, fail = 0, thrown = 0;
    for (auto & c : cases) {
        common_chat_templates_inputs in;
        common_chat_msg user; user.role = "user"; user.content = "go";
        in.messages = { user };
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        in.add_generation_prompt = true;
        in.parallel_tool_calls = true;
        for (auto & t : c["tools"]) {
            in.tools.push_back({ t["name"].get<std::string>(), t["description"].get<std::string>(), t["parameters"].dump() });
        }
        std::vector<std::string> got;
        std::string content, reasoning, err;
        try {
            auto params = common_chat_templates_apply(tmpls.get(), in);
            common_peg_arena arena; arena.load(params.parser);
            common_chat_parser_params pp(params);
            pp.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            std::string input = c["input"].get<std::string>();
            // streaming pass (width 7): must never throw
            for (size_t i = 7; i < input.size(); i += 7) { (void) common_chat_peg_parse(arena, input.substr(0, i), true, pp); }
            auto msg = common_chat_peg_parse(arena, input, c["partial"].get<bool>(), pp);
            for (auto & tc : msg.tool_calls) { got.push_back(tc.name); }
            content = msg.content; reasoning = msg.reasoning_content;
        } catch (std::exception & e) { err = e.what(); }
        std::vector<std::string> want = c["expect_calls"].get<std::vector<std::string>>();
        bool ok = err.empty() && got == want;
        if (!err.empty()) { thrown++; }
        (ok ? pass : fail)++;
        auto join = [](const std::vector<std::string> & v) { std::string s; for (auto & x : v) { s += (s.empty() ? "" : ",") + x; } return "[" + s + "]"; };
        std::cout << (ok ? "PASS " : "FAIL ") << c["name"].get<std::string>() << "  want=" << join(want) << " got=" << join(got);
        if (!err.empty()) { std::cout << " THROW: " << err.substr(0, 120); }
        std::cout << "\n";
        if (verbose && !ok) {
            std::cout << "   content:   " << content.substr(0, 160) << "\n   reasoning: " << reasoning.substr(0, 160) << "\n";
        }
    }
    std::cout << "\n" << pass << " pass, " << fail << " fail (" << thrown << " threw) of " << cases.size() << "\n";
    return 0;
}
