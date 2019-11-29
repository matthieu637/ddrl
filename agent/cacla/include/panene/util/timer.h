#ifndef TIMER_H
#define TIMER_H

#ifdef _WIN32 
#include <chrono>
#else
#include <sys/time.h>
#include <ctime>
#endif

namespace panene {

    class Timer
    {
    public:
        Timer() {}

        /* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
        * windows and linux. */

        typedef long long int64;
        typedef unsigned long long uint64;

#ifndef _WIN32
        uint64 GetTimeMs64()
        {
            /* Linux */
            struct timeval tv;

            gettimeofday(&tv, NULL);

            uint64 ret = tv.tv_usec;
            /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
            ret /= 1000;

            /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
            ret += (tv.tv_sec * 1000);

            return ret;
        }
#endif

        void begin() {
#ifdef _WIN32
            begint = std::chrono::steady_clock::now();
#else
            bb = GetTimeMs64();
#endif
        }

        double end() {
#ifdef _WIN32
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            return (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begint).count() / 1000000;
#else
            ff = GetTimeMs64();

            double elapsed = (ff - bb) / 1000.0;
            return elapsed;
#endif
        }

    protected:

#ifdef _WIN32
        std::chrono::steady_clock::time_point begint;
#else
        uint64 bb, ff;
#endif
    };

}

#endif
