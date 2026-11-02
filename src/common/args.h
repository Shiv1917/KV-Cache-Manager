// Tiny "--key=value" / "--key value" / "--flag" command-line parser.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace kvp {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) throw std::invalid_argument("bad argument: " + a);
      a = a.substr(2);
      auto eq = a.find('=');
      if (eq != std::string::npos) {
        kv_[a.substr(0, eq)] = a.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[a] = argv[++i];
      } else {
        kv_[a] = "1";
      }
    }
  }
  bool has(const std::string& k) const { used_.insert(k); return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def) const {
    used_.insert(k);
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  long long i64(const std::string& k, long long def) const {
    auto s = str(k, "");
    return s.empty() ? def : std::stoll(s);
  }
  int i32(const std::string& k, int def) const { return static_cast<int>(i64(k, def)); }
  double f64(const std::string& k, double def) const {
    auto s = str(k, "");
    return s.empty() ? def : std::stod(s);
  }
  bool flag(const std::string& k, bool def = false) const {
    auto s = str(k, "");
    if (s.empty()) return def;
    return s == "1" || s == "true" || s == "on" || s == "yes";
  }
  // Warn about flags nobody read (typos).
  void warn_unused() const {
    for (auto& [k, v] : kv_)
      if (!used_.count(k)) std::fprintf(stderr, "warning: unused flag --%s\n", k.c_str());
  }

 private:
  std::map<std::string, std::string> kv_;
  mutable std::set<std::string> used_;
};

}  // namespace kvp
