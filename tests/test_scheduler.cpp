#include <gtest/gtest.h>
#include "scheduler/thread_pool.h"
#include <atomic>

using namespace infer;

TEST(SchedulerTest, ThreadPoolExecution) {
    ThreadPool pool(2);
    std::atomic<int> counter(0);
    
    for (int i = 0; i < 10; ++i) {
        pool.enqueue([&counter]() {
            counter++;
        });
    }
    
    pool.wait();
    EXPECT_EQ(counter.load(), 10);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
