#pragma once
#include <mutex>
#include <condition_variable>
#include <list>

namespace fdm {

template<typename T>
struct concurrent_queue {
    std::list<T> queue;
    std::mutex m;
    std::condition_variable cv;

    void push(T&& task) {
        std::unique_lock lock(m);
        queue.emplace_back(std::move(task));
        cv.notify_one();
    }

    T pop() {
        std::unique_lock lock(m);
        while (queue.empty()) {
            cv.wait(lock);
        }
        T ret = std::move(queue.front());
        queue.pop_front();
        return ret;
    }
};

} // namespace fdm
