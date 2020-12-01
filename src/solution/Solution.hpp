#pragma once

#include <map>
#include <memory>
#include <sstream>

namespace aoc {

// A self registering solution base class
// Reference:
// https://stackoverflow.com/questions/11175379/register-a-c-class-so-that-later-a-function-can-iterate-over-all-registered-cl
class Solution {
  public:
    using SolPtr = std::unique_ptr<Solution>;

    template <class T> class Registrar {
      public:
        explicit Registrar(int year, int day, std::string description) {
            auto newSol = T::create();
            newSol->description = description;
            Solution::registrate(year, day, std::move(newSol));
        }
    };

    virtual ~Solution() = default;

    static std::unique_ptr<Solution> instantiate(int year, int day) {
        auto it = registry().find(getNameKey(year, day));
        return it == registry().end() ? nullptr : std::move(it->second);
    }

    virtual void Calculate(std::istream& input) = 0;

    std::string description;

  protected:
    static void registrate(int year, int day, SolPtr fp) {
        registry()[getNameKey(year, day)] = std::move(fp);
    }

  private:
    static std::map<std::string, SolPtr>& registry();

    static std::string getNameKey(int year, int day) {
        std::stringstream ss;
        ss << year << "-" << day;
        return ss.str();
    }
};

} // namespace aoc
