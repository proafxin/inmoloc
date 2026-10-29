#include "server-task.h"
#include "server-queue.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <thread>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

static bool task_resets_idle_timer(server_task_type type) {
    return type != SERVER_TASK_TYPE_METRICS;
}

int server_queue::post(server_task && task, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target);
    }
    if (task.t_arrival_us == 0) {
        task.t_arrival_us = ggml_time_us();
    }
    const int  task_id     = task.id;
    const bool reset_timer = task_resets_idle_timer(task.type);
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    if (reset_timer) {
        time_last_task = ggml_time_ms();
    }
    condition_tasks.notify_one();
    return task_id;
}

int server_queue::post(std::vector<server_task> && tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    bool reset_timer = false;
    const int64_t t_now_us = ggml_time_us();
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        if (task.t_arrival_us == 0) {
            task.t_arrival_us = t_now_us;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target);
        }
        reset_timer |= task_resets_idle_timer(task.type);
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    if (reset_timer) {
        time_last_task = ggml_time_ms();
    }
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task && task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    QUE_DBG("defer task, id = %d\n", task.id);
    queue_tasks_deferred.push_back(std::move(task));
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

// a task that waits for memory (KV cells, a recurrent-state cell) is offered again only by pop_deferred_tasks(ids),
// once the memory it needs is free
static bool waits_for_memory(const server_task & task) {
    return task.defer_reason == SERVER_DEFER_KV_CELLS || task.defer_reason == SERVER_DEFER_RECURRENT_CELL;
}

void server_queue::pop_deferred_task(int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    // try to find a task that uses the specified slot
    auto it = std::find_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), [&](const server_task & task) {
        return task.id_slot == id_slot && !waits_for_memory(task);
    });
    // if not tasks found using the slot, just pop the first deferred task (default behavior)
    if (it == queue_tasks_deferred.end()) {
        it = std::find_if_not(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), waits_for_memory);
    }
    if (it != queue_tasks_deferred.end()) {
        QUE_DBG("pop deferred task, id_task = %d\n", it->id);
        queue_tasks.emplace_front(std::move(*it));
        queue_tasks_deferred.erase(it);
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

std::vector<server_queue::mem_wait> server_queue::get_deferred_mem_tasks() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    std::vector<mem_wait> res;
    for (const auto & task : queue_tasks_deferred) {
        if (waits_for_memory(task)) {
            res.push_back({ task.id, task.n_kv_need, task.n_kv_wait_at, task.n_rs_need });
        }
    }
    return res;
}

void server_queue::pop_deferred_tasks(const std::vector<int> & ids) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    std::vector<server_task> tasks;
    for (const int id : ids) {
        // a task cancelled since it was listed is gone
        auto it = std::find_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), [id](const server_task & task) {
            return task.id == id;
        });
        if (it != queue_tasks_deferred.end()) {
            tasks.push_back(std::move(*it));
            queue_tasks_deferred.erase(it);
        }
    }
    if (tasks.empty()) {
        return;
    }
    QUE_DBG("pop %zu deferred tasks\n", tasks.size());
    for (size_t i = tasks.size(); i-- > 0;) {
        queue_tasks.emplace_front(std::move(tasks[i]));
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

void server_queue::wait_until_no_sleep() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleeping) {
        return;
    } else {
        if (!req_stop_sleeping) {
            QUE_DBG("%s", "requesting to stop sleeping\n");
            req_stop_sleeping = true;
            condition_tasks.notify_one(); // only main thread is waiting on this
        }
        QUE_DBG("%s", "waiting until no sleep\n");
        condition_tasks.wait(lock, [&]{
            return !sleeping;
        });
    }
}

void server_queue::terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    condition_tasks.notify_all();
}

bool server_queue::process_new_tasks(bool is_yielding) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!running) {
            QUE_DBG("%s", "terminate\n");
            return true;
        }
        if (queue_tasks.empty()) {
            return false;
        }
        server_task task = std::move(queue_tasks.front());
        queue_tasks.pop_front();
        lock.unlock();

        QUE_DBG("processing task, id = %d\n", task.id);
        if (!callback_new_task(std::move(task), is_yielding)) {
            // set it aside, do not put it back in the queue, else we offer it again in a loop
            GGML_ASSERT(is_yielding && "a task can only be declined while yielding");
            QUE_DBG("task declined, id = %d\n", task.id);
            lock.lock();
            queue_tasks_unhandled.push_back(std::move(task));
        }
    }
}

void server_queue::worker_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            // wait on busy instead of yielding - busy stays set even when the yield already ended
            worker.cv.wait(lock, [&]{
                return worker.stop || worker.busy;
            });
            if (worker.stop) {
                return;
            }
        }

        // process tasks while the yield is active
        while (true) {
            bool terminated = false;
            try {
                // note: do not hold any lock here, the callback may post new tasks
                terminated = process_new_tasks(true);
            } catch (...) {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                worker.exception = std::current_exception();
                break;
            }

            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (terminated || worker.stop || !worker.yielding) {
                break;
            }
            if (!queue_tasks.empty()) {
                continue; // a new task arrived in the meantime
            }
            condition_tasks.wait(lock, [&]{
                return worker.stop || !running || !worker.yielding || !queue_tasks.empty();
            });
        }

        // signal to yield_to_queue() that no more tasks will be processed
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            worker.busy = false;
        }
        condition_tasks.notify_all();
    }
}

void server_queue::worker_stop() {
    if (!worker.thread.joinable()) {
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        worker.stop = true;
    }
    worker.cv.notify_one();
    condition_tasks.notify_all();
    worker.thread.join();
}

void server_queue::yield_to_queue(std::function<void()> && work) {
    GGML_ASSERT(worker.thread.joinable() && "yield_to_queue() requires start_loop() to be running");

    QUE_DBG("%s", "yielding to queue\n");

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        GGML_ASSERT(!worker.busy && "yield_to_queue() cannot be nested");
        worker.busy     = true;
        worker.yielding = true;
    }
    worker.cv.notify_one();

    // run the work on the current thread, so that all ggml compute stays on the same thread
    std::exception_ptr exception;
    try {
        work();
    } catch (...) {
        exception = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(mutex_tasks);

        // the yield is over, wait for the worker to finish its current task
        worker.yielding = false;
        condition_tasks.notify_all();
        condition_tasks.wait(lock, [&]{
            return !worker.busy;
        });

        // put the declined tasks back, keeping their order
        while (!queue_tasks_unhandled.empty()) {
            queue_tasks.push_front(std::move(queue_tasks_unhandled.back()));
            queue_tasks_unhandled.pop_back();
        }

        // make sure to avoid idle timeout here
        time_last_task = ggml_time_ms();

        // an exception from work() takes precedence over the one from the worker
        if (!exception) {
            std::swap(exception, worker.exception);
        } else {
            worker.exception = nullptr;
        }
    }

    QUE_DBG("%s", "done yielding to queue\n");

    // note: rethrow only after the declined tasks are back in the queue, so they are not lost
    if (exception) {
        std::rethrow_exception(exception);
    }
}

void server_queue::start_loop(int64_t idle_sleep_ms) {
    running = true;
    time_last_task = ggml_time_ms();

    // spawn the worker thread used by yield_to_queue()
    GGML_ASSERT(!worker.thread.joinable() && "start_loop() is already running");
    worker.stop     = false;
    worker.busy     = false;
    worker.yielding = false;
    worker.thread = std::thread([this]() { worker_loop(); });

    constexpr auto max_wait_time = std::chrono::seconds(1);
    auto should_sleep = [&]() -> bool {
        // caller must hold mutex_tasks
        if (idle_sleep_ms < 0) {
            return false;
        }
        int64_t now = ggml_time_ms();
        return (now - time_last_task) >= idle_sleep_ms;
    };

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");
        if (process_new_tasks(false)) {
            break; // terminate
        }

        // all tasks in the current loop is processed, slots data is now ready
        QUE_DBG("%s", "update slots\n");

        // this will run the main inference process for all slots
        const int64_t t_update_slots = ggml_time_ms();
        callback_update_slots();
        {
            // update_slots() may take a while to finish, we need to make sure it's not counted as idle
            // shift instead of reset, so that non-task_resets_idle_timer tasks do not delay the sleep
            std::unique_lock<std::mutex> lock(mutex_tasks);
            const int64_t now = ggml_time_ms();
            time_last_task = std::min(now, time_last_task + (now - t_update_slots));
        }

        QUE_DBG("%s", "waiting for new tasks\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running || !queue_tasks.empty()) {
                break; // go back to process new tasks or terminate
            }

            // no tasks, check for sleeping state
            if (should_sleep()) {
                QUE_INF("%s", "entering sleeping state\n");
                sleeping = true;
                // Call order cb0 -> cb1 -> cb{N}
                for (auto & cb : callback_sleeping_state) {
                    cb(true);
                }
                req_stop_sleeping = false;
                // wait until we are requested to exit sleeping state
                condition_tasks.wait(lock, [&]{
                    return (!running || req_stop_sleeping);
                });
                if (!running) { // may changed during sleep
                    break; // terminate
                }
                QUE_INF("%s", "exiting sleeping state\n");
                req_stop_sleeping = false;
                // Call order cb{N} -> cb1 -> cb0
                for (size_t i = callback_sleeping_state.size(); i > 0; i--) {
                    callback_sleeping_state[i - 1](false);
                }
                sleeping = false;
                time_last_task = ggml_time_ms();
                condition_tasks.notify_all(); // notify wait_until_no_sleep()
                break; // process new tasks
            } else {
                // wait for new tasks or timeout for checking sleeping condition
                bool res = condition_tasks.wait_for(lock, max_wait_time, [&]{
                    return (!queue_tasks.empty() || !running);
                });
                if (res) {
                    break; // new task arrived or terminate
                }
                // otherwise, loop again to check sleeping condition
            }
        }
    }

    worker_stop();
}

void server_queue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task & task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(),           queue_tasks.end(),           rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(),  queue_tasks_deferred.end(),  rm_func),
        queue_tasks_deferred.end());
    // a task declined while yielding is not in queue_tasks yet, but it can still be cancelled
    queue_tasks_unhandled.erase(
        std::remove_if(queue_tasks_unhandled.begin(), queue_tasks_unhandled.end(), rm_func),
        queue_tasks_unhandled.end());
}

//
// server_response
//

std::shared_ptr<server_response::waiter> server_response::find_waiter(int id_task) {
    std::unique_lock<std::mutex> lock(mutex_waiters);
    const auto it = waiters.find(id_task);
    return it == waiters.end() ? nullptr : it->second;
}

std::shared_ptr<server_response::waiter> server_response::waiter_for(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_waiters);
    for (const auto & id_task : id_tasks) {
        const auto it = waiters.find(id_task);
        if (it != waiters.end()) {
            return it->second;
        }
    }
    return std::make_shared<waiter>();
}

server_task_result_ptr server_response::take_result(waiter & w, const std::unordered_set<int> & id_tasks) {
    for (auto it = w.results.begin(); it != w.results.end(); ++it) {
        if (id_tasks.find((*it)->id) != id_tasks.end()) {
            server_task_result_ptr res = std::move(*it);
            w.results.erase(it);
            return res;
        }
    }
    return nullptr;
}

void server_response::add_waiting_task_id(int id_task) {
    add_waiting_task_ids({ id_task });
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    // the tasks of one reader share its waiter, so that it waits on one condition variable for all of them
    auto w = waiter_for(id_tasks);

    std::unique_lock<std::mutex> lock(mutex_waiters);
    for (const auto & id_task : id_tasks) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiters.size());
        waiters.emplace(id_task, w);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    remove_waiting_task_ids({ id_task });
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::vector<std::shared_ptr<waiter>> removed;
    {
        std::unique_lock<std::mutex> lock(mutex_waiters);
        for (const auto & id_task : id_tasks) {
            RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiters.size());
            const auto it = waiters.find(id_task);
            if (it != waiters.end()) {
                removed.push_back(it->second);
                waiters.erase(it);
            }
        }
    }

    // make sure to clean up all pending results
    for (const auto & w : removed) {
        std::unique_lock<std::mutex> lock(w->mutex);
        w->results.erase(
            std::remove_if(w->results.begin(), w->results.end(), [&](const server_task_result_ptr & res) {
                return id_tasks.find(res->id) != id_tasks.end();
            }),
            w->results.end());
    }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    auto w = waiter_for(id_tasks);

    std::unique_lock<std::mutex> lock(w->mutex);
    while (true) {
        if (!running) {
            RES_DBG("%s : queue result stop\n", "recv");
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (auto res = take_result(*w, id_tasks)) {
            return res;
        }
        w->cond.wait(lock);
    }
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    auto w = waiter_for(id_tasks);

    std::unique_lock<std::mutex> lock(w->mutex);
    while (true) {
        if (auto res = take_result(*w, id_tasks)) {
            return res;
        }

        std::cv_status cr_res = w->cond.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            RES_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return take_result(*w, id_tasks);
        }
    }
}

server_task_result_ptr server_response::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);

    // a result for a task nobody waits for is dropped
    auto w = find_waiter(result->id);
    if (!w) {
        return;
    }

    RES_DBG("task id = %d pushed to result queue\n", result->id);
    {
        std::unique_lock<std::mutex> lock(w->mutex);
        w->results.emplace_back(std::move(result));
    }
    w->cond.notify_all();
}

void server_response::broadcast(server_task_result_ptr && result) {
    std::vector<std::pair<int, std::shared_ptr<waiter>>> targets;
    {
        std::unique_lock<std::mutex> lock(mutex_waiters);
        targets.assign(waiters.begin(), waiters.end());
    }
    for (const auto & [id_task, w] : targets) {
        RES_DBG("task id = %d pushed to result queue\n", id_task);
        server_task_result_ptr res_copy(result->clone());
        res_copy->id = id_task; // override id with target task id
        {
            std::unique_lock<std::mutex> lock(w->mutex);
            w->results.emplace_back(std::move(res_copy));
        }
        w->cond.notify_all();
    }
}

void server_response::terminate() {
    running = false;

    std::vector<std::shared_ptr<waiter>> all;
    {
        std::unique_lock<std::mutex> lock(mutex_waiters);
        for (const auto & [id_task, w] : waiters) {
            all.push_back(w);
        }
    }
    for (const auto & w : all) {
        // take the waiter's lock so that a reader between its check and its wait does not miss the notification
        std::unique_lock<std::mutex> lock(w->mutex);
        w->cond.notify_all();
    }
}

//
// server_response_reader
//

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    queue_tasks.post(std::move(task), front);
}

void server_response_reader::post_tasks(std::vector<server_task> && tasks, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
    id_tasks = server_task::get_list_id(tasks);
    states.reserve(tasks.size());
    size_t index = 0;
    for (auto & task : tasks) {
        task.index = index++;
        states.push_back(task.create_state());
        // for child tasks
        for (auto & child_task : task.child_tasks) {
            child_task.index = index++;
            states.push_back(child_task.create_state());
        }
    }
    GGML_ASSERT(states.size() == id_tasks.size());
    queue_results.add_waiting_task_ids(id_tasks);
    queue_tasks.post(std::move(tasks), front);
}

bool server_response_reader::has_next() const {
    return !cancelled && received_count < id_tasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
        if (result == nullptr) {
            // timeout, check stop condition
            if (should_stop()) {
                return nullptr;
            }
        } else {
            if (result->is_error()) {
                stop(); // cancel remaining tasks
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
    }

    // should not reach here
}

server_response_reader::batch_response server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response batch_res;
    batch_res.results.clear();
    batch_res.results.resize(id_tasks.size());
    while (has_next()) {
        auto res = next(should_stop);
        if (res == nullptr) {
            batch_res.is_terminated = true;
            return batch_res;
        }
        if (res->is_error()) {
            batch_res.error = std::move(res);
            return batch_res;
        }
        const size_t idx = res->index;
        GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
        GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
        batch_res.results[idx] = std::move(res);
    }
    return batch_res;
}

void server_response_reader::stop() {
    queue_results.remove_waiting_task_ids(id_tasks);
    if (has_next() && !cancelled) {
        // if tasks is not finished yet, cancel them
        cancelled = true;
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(std::move(task));
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(std::move(cancel_tasks), true);
    } else {
        SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
    }
}
