#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

// define the class

class ThreadPool {
public:
    // n of threads to create
    // sets the control variable to false
    explicit ThreadPool(std::size_t threads) : stop(false) {
        for (std::size_t i = 0; i < threads; ++i) { // for every thread requested
            // adds thread to the end of the workers vector
            workers.emplace_back([this]() { // assigning it the code to execute in the form of a lambda 
                // worker cycle life
                while (true) {
                    std::function<void()> task; // prepare his task

                    // open block
                    {   // locks the queue so only one thread can touch it at a time
                        std::unique_lock<std::mutex> lock(this->queue_mutex); 
                        // put the thread to sleep unitll
                        this->condition.wait(lock, [this]() {
                            // the pool is being closed
                            // there is at least one job in the queue
                            return this->stop || !this->tasks.empty();
                        });
                        
                        // if the pool is stopping and no jobs remain in the queue -> return
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        // move the front job out (no copy) and pop it from the queue
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    // close block

                    task(); // run the job outside the lock in parallel with other workers
                }
            });
        }
    }

    // destroyer of the class

    ~ThreadPool() {

        // open block
        {
            // takes the lock
            std::unique_lock<std::mutex> lock(queue_mutex);
            // set the flag true
            stop = true;
        }
        // close block
        
        // wakes up all waiting threads
        // stop is true they exit as soon as the queue is empty
        condition.notify_all();

        for (std::thread& worker : workers) { // for every thread in the workers vec
            if (worker.joinable()) { // if it's still joinable
                worker.join(); // wait for it to finish avoiding orphan threads
            }
        }
    }

    // enqueue entry point for assigning work

    template<class F, class... Args> // makes enqueue a variadic template

    // deduces the return type of f when called with args
    // enqueue takes a callable f and its args,
    // returning a future that will hold the result of f(args)
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using return_type = typename std::invoke_result_t<F, Args...>; // alias for f's return type

        // packaged_task links execution to a futur
        // shared_ptr since it's not copyable
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            // freeze f + args into a zero-arg callable
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        // grab the future linked to this task, to return to the caller
        std::future<return_type> res = task->get_future();
        
        // open block
        {   
            // takes the lock
            std::unique_lock<std::mutex> lock(queue_mutex);
            // check if is closing
            if (stop) {
                throw std::runtime_error("enqueue call on a stopping ThreadPool"); // error
            }
            tasks.emplace([task]() { (*task)(); }); // wrap task in a lambda and push it to the queue
        }
        // close block
        
        condition.notify_one(); // wake one worker to pick up the new task
        return res; // caller gets the future now result comes later via res.get()
    }

// private 

private:
    std::vector<std::thread> workers; // dynamic array that stores the thread objects created by the constructor
    std::queue<std::function<void()>> tasks; // queue FIFO where waiting jobs are held

    std::mutex queue_mutex; // lock
    std::condition_variable condition; // signaling mechanism
    bool stop; // flag
};
