#ifndef LOGGER_HPP
#define LOGGER_HPP

///
// /\file Logger.hpp
// /\brief Singleton de Logger afin d'écrire des logs
// /\example Dans LoggerTest.hpp
///
///

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <list>
#include <map>
#include <utility>

#include "bib/Assert.hpp"
#include "Singleton.hpp"

#define LOG(stream)                                            \
  bib::Logger::getInstance()->isBufferEnable()                 \
      ? bib::Logger::getInstance()->registerBuffer() << stream \
      : std::cout << stream << std::endl

#define LOG_DEBUG(stream)                                             \
  bib::Logger::getInstance()->isEnabled(bib::Logger::DEBUGGING) &&    \
      std::cout << "#DEBUG :" << __FILE__ << "." << __LINE__ << " : " \
                << stream << std::endl

#define LOG_DEBUGS(stream)                                            \
  bib::Logger::getInstance()->isEnabled(bib::Logger::DEBUGGING) &&    \
      std::cout << "#DEBUG :" << __FILE__ << "." << __LINE__ << " : " \
                << stream

#define LOG_DEBUGC(stream)                                         \
  bib::Logger::getInstance()->isEnabled(bib::Logger::DEBUGGING) && \
      std::cout << stream

#define LOG_DEBUGE(stream)                                         \
  bib::Logger::getInstance()->isEnabled(bib::Logger::DEBUGGING) && \
      std::cout << std::endl

#define LOG_INFO(stream)                                                       \
  bib::Logger::getInstance()->isEnabled(bib::Logger::INFO) &&                  \
      std::cout << "#INFO :" << __FILE__ << "." << __LINE__ << " : " << stream \
                << std::endl

#define LOG_WARNING(stream)                                             \
  bib::Logger::getInstance()->isEnabled(bib::Logger::WARNING) &&        \
      std::cout << "#WARNING :" << __FILE__ << "." << __LINE__ << " : " \
                << stream << std::endl

#define LOG_ERROR(stream)                                             \
  bib::Logger::getInstance()->isEnabled(bib::Logger::ERROR) &&        \
      std::cout << "#ERROR :" << __FILE__ << "." << __LINE__ << " : " \
                << stream << std::endl

#define LOG_FILE(file, stream) \
  bib::Logger::getInstance()->getFile(file) << stream << std::endl;

namespace bib {

class Logger : public Singleton<Logger> {
  friend class Singleton<Logger>;
  friend class SingletonFactory;

 public:
  enum LogLevel { DEBUGGING, INFO, WARNING, ERROR };

  std::ofstream &getFile(const std::string &s) {
    if (open_files.find(s) == open_files.end()) {
      std::ofstream *ofs = new std::ofstream;
      ofs->open(s, std::ofstream::out);
      open_files.insert(std::pair<std::string, std::ofstream *>(s, ofs));
    }

    return *open_files.find(s)->second;
  }
  std::stringstream &registerBuffer() {
    std::stringstream *n = new std::stringstream;

    if (!ignored_buffer_enable)
      buff.push_back(n);
    else
      ignored_buffer.push_back(n);

    return *n;
  }

  void flushBuffer() {
    for (std::list<std::stringstream *>::iterator it = buff.begin();
         it != buff.end(); ++it) {
      std::cout << (*it)->str() << " ";
      delete *it;
    }
    std::cout << std::endl;
    buff.clear();

    clearIgnoredBuffer();
  }

  void clearIgnoredBuffer() {
    for (std::list<std::stringstream *>::iterator it = ignored_buffer.begin();
         it != ignored_buffer.end(); ++it)
      delete *it;
    ignored_buffer.clear();
  }

  void enableBuffer() {
    enable_buffer = true;
  }

  bool isEnabled(LogLevel l) const {
    return level <= l;
  }

  bool isBufferEnable() const {
    return enable_buffer;
  }

  void setLevel(LogLevel l) {
    level = l;
  }

  void setIgnoredBuffer(bool val) {
    ignored_buffer_enable = val;
  }

  template <class T>
  static inline void PRINT_ELEMENTS(const T &coll, const char *optcstr = "") {
    typename T::const_iterator pos;

    LOG_DEBUGS(optcstr);
    for (pos = coll.begin(); pos != coll.end(); ++pos)
      LOG_DEBUGC(*pos << ",\t");

    LOG_DEBUGE();
  }

  template <class T>
  static inline void PRINT_ELEMENTS_FT(const T &coll, const char *optcstr = "",
                                       int width = 4, int precision = 2) {
    typename T::const_iterator pos;

    LOG_DEBUGS(optcstr);
    for (pos = coll.begin(); pos != coll.end(); ++pos)
      LOG_DEBUGC(std::left << std::setw(width) << std::setfill(' ')
                 << std::setprecision(precision) << *pos);

    LOG_DEBUGE();
  }

  template <class T>
  static inline void PRINT_ELEMENTS(const T &coll, int length,
                                    const char *optcstr = "") {
    LOG_DEBUGS(optcstr);
    for (int i = 0; i < length; i++) LOG_DEBUGC(coll[i] << ", ");

    LOG_DEBUGE();
  }

  ~Logger() {
    clearIgnoredBuffer();
    for (std::list<std::stringstream *>::iterator it = buff.begin();
         it != buff.end(); ++it)
      delete *it;
    buff.clear();

    for (auto it = open_files.begin(); it != open_files.end(); ++it) {
      it->second->close();
      delete it->second;
    }
  }

 protected:
  Logger() : level(DEBUGGING) {}

 private:
  LogLevel level;

  bool enable_buffer = false;
  bool ignored_buffer_enable = false;
  std::list<std::stringstream *> buff;
  std::list<std::stringstream *> ignored_buffer;
  unsigned int buffer_index = 0;
  std::map<std::string, std::ofstream *> open_files;
};
}  // namespace bib

#endif
