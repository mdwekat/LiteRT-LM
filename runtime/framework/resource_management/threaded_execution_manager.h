// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_

#include <atomic>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {

// Threaded implementation of the ExecutionManager.
// This implementation uses a thread pool for task execution and another one for
// callbacks.
class ThreadedExecutionManager : public ExecutionManager {
 public:
  // Creates a new ThreadedExecutionManager.
  static absl::StatusOr<std::unique_ptr<ThreadedExecutionManager>> Create(
      Tokenizer* absl_nonnull tokenizer,
      ModelResources* absl_nullable model_resources,
      std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
      std::unique_ptr<VisionExecutorSettings> absl_nullable
      vision_executor_settings,
      std::unique_ptr<AudioExecutorSettings> absl_nullable
      audio_executor_settings,
      ::litert::Environment* absl_nullable litert_env,
      std::unique_ptr<AudioExecutor> absl_nullable audio_executor = nullptr);

  ~ThreadedExecutionManager() override;

  // Waits until the task is done or the timeout is reached.
  // Returns:
  // - OK if the task is done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if the task is failed.
  absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Waits until all tasks in the session are done or the timeout is reached.
  absl::Status WaitUntilSessionDone(SessionId session_id,
                                    absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Waits until all tasks are done or the timeout is reached.
  // Returns:
  // - OK if all tasks are done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if any of the tasks is failed.
  absl::Status WaitUntilAllDone(absl::Duration timeout) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns a new session ID.
  // The returned session ID is guaranteed to be unique.
  absl::StatusOr<SessionId> RegisterNewSession(
      SessionConfig session_config,
      std::optional<BenchmarkInfo> benchmark_info) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Releases the session with the given session ID.
  absl::Status ReleaseSession(SessionId session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Cancels all tasks in the session with the given session ID.
  absl::Status CancelAllTasksInSession(SessionId session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::StatusOr<std::shared_ptr<const SessionInfo>> GetSessionInfo(
      SessionId session_id) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo(SessionId session_id)
      override ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::StatusOr<TaskId> GetNewTaskId() override;

  // Adds a prefill task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - inputs: The inputs of the prefill task.
  // - dep_tasks: The dependent tasks that should be done before the prefill
  //   task starts.
  // - cancelled: The cancelled flag for the prefill task.
  // - callback: The callback function.
  // Note: AddPrefillTask will acquire the task lookup mutex.
  absl::Status AddPrefillTask(
      SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
      absl::flat_hash_set<TaskId> dep_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a decode task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the decode
  //   task starts.
  // - constraint: The constraint for the decode task.
  // - cancelled: The cancelled flag for the decode task.
  // - callback: The callback function.
  // - max_output_tokens: The maximum number of tokens to decode.
  // Note: AddDecodeTask will acquire the task lookup mutex.
  absl::Status AddDecodeTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      int max_output_tokens) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a clone session task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the clone
  //   session task starts.
  // - cloned_session_id: The ID of the cloned session.
  // - callback: The callback function.
  // Note: AddCloneSessionTask will acquire the task lookup mutex.
  // TODO b/409401231 - Add unit tests for this function.
  absl::Status AddCloneSessionTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks, SessionId cloned_session_id,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Adds a text scoring task to the execution manager.
  // - session_id: The ID of the session that created the task.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the text
  //   scoring task starts.
  // - target_text: The target text to be scored.
  // - store_token_lengths: Whether to store the token lengths in the
  //   responses.
  // - cancelled: The cancelled flag for the text scoring task.
  // - callback: The callback function.
  // Note: AddTextScoringTask will acquire the task lookup mutex.
  absl::Status AddTextScoringTask(
      SessionId session_id, TaskId task_id,
      absl::flat_hash_set<TaskId> dep_tasks,
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  absl::StatusOr<int> GetCurrentStep(const SessionInfo& session_info) override;

  absl::Status SetCurrentStep(const SessionInfo& session_info,
                              int target_step) override;

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override;

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override;

 private:
  ThreadedExecutionManager(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
      ::litert::Environment* absl_nullable litert_env = nullptr);

  // Creates a task and adds it to the task list.
  // Note: This method will acquire the task lookup mutex.
  absl::Status CreateTask(
      SessionId session_id, TaskId task_id,
      absl::AnyInvocable<void()> absl_nonnull task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Queues the task with the given task ID to the execution thread pool.
  absl::Status QueueTask(TaskId task_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Starts the task with the given task ID.
  // Returns:
  // - A tuple containing the session info, the cancelled flag, and the
  //   callback function.
  // - INVALID_ARGUMENT if the task ID is not found.
  // - FAILED_PRECONDITION if the task is not in the queued state.
  absl::StatusOr<std::tuple<
      std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTask(TaskId task_id) ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Finishes the task with the given task ID.
  // This method will handle the task state transition and call the callback
  // in the callback thread pool.
  absl::Status FinishTask(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Finishes the task with the given task ID and logs errors if any.
  void FinishTaskAndLogErrors(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(session_and_task_lookup_mutex_);

  // Returns the following tasks that are waiting for the given task to finish.
  absl::StatusOr<absl::flat_hash_set<TaskId>> FollowingWaitingTasks(
      TaskId task_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Updates the state of the task with the given task ID.
  absl::Status UpdateTaskState(TaskId task_id, TaskState task_state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  // Updates the state of all the given tasks to the given task state.
  absl::Status UpdateAllTasksToState(
      const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_);

  absl::StatusOr<ExecutorInputs> ProcessAndCombineContents(
      const std::vector<InputData>& preprocessed_contents,
      std::optional<BenchmarkInfo>& benchmark_info);

  // The next session ID to be returned.
  std::atomic<SessionId> next_session_id_ = 0;
  // The next task ID to be returned.
  std::atomic<TaskId> next_task_id_ = 0;
  // Mutex to protect the session and task lookup.
  absl::Mutex session_and_task_lookup_mutex_;
  // Mapping from session ID to session info.
  absl::flat_hash_map<SessionId, std::shared_ptr<SessionInfo> absl_nonnull>
      session_lookup_ ABSL_GUARDED_BY(session_and_task_lookup_mutex_) = {};
  // Mapping from task ID to task info.
  absl::flat_hash_map<TaskId, TaskInfo> task_lookup_
      ABSL_GUARDED_BY(session_and_task_lookup_mutex_) = {};
  // The last prefill token ID.
  int last_prefill_token_id_ = 0;
  // Tokenizer used for tokenization and detokenization.
  Tokenizer* absl_nonnull tokenizer_;
  // Resource manager used for managing the model resources and executors.
  std::unique_ptr<ResourceManager> absl_nonnull resource_manager_;
  // LiteRT environment used for creating the sampler.
  ::litert::Environment* absl_nullable litert_env_;
  // Thread pool used for executing the tasks.
  std::unique_ptr<ThreadPool> absl_nonnull execution_thread_pool_;
  // Thread pool used for executing the callbacks.
  std::unique_ptr<ThreadPool> absl_nonnull callback_thread_pool_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_THREADED_EXECUTION_MANAGER_H_
