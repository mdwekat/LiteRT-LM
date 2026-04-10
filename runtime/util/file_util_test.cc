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

#include "runtime/util/file_util.h"

#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

#if defined(_WIN32)
constexpr absl::string_view kPathSeparator = "\\";
#else
constexpr absl::string_view kPathSeparator = "/";
#endif

TEST(FileUtilTest, JoinPath) {
  std::string path1 = "";
  std::string path2 = "path2";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::InvalidArgumentError("Empty path1."));

  path1 = "path1";
  path2 = "";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::InvalidArgumentError("Empty path2."));

  path1 = "path1";
  path2 = "path2";
  EXPECT_THAT(JoinPath(path1, path2),
              absl::StrCat("path1", kPathSeparator, "path2"));
}

TEST(FileUtilTest, Basename) {
  std::string model_path = absl::StrCat(kPathSeparator, "path", kPathSeparator,
                                        "to", kPathSeparator, "model.tflite");
  EXPECT_THAT(Basename(model_path), "model.tflite");
}

TEST(FileUtilTest, Dirname) {
  std::string model_path = absl::StrCat(kPathSeparator, "path", kPathSeparator,
                                        "to", kPathSeparator, "model.tflite");
  EXPECT_THAT(Dirname(model_path),
              absl::StrCat(kPathSeparator, "path", kPathSeparator, "to",
                           kPathSeparator));
}

TEST(FileUtilTest, GetFileCacheIdentifier) {
  ASSERT_OK_AND_ASSIGN(auto temp_file,
                       JoinPath(testing::TempDir(), "test_file.txt"));
  std::ofstream ofs(temp_file);
  ofs << "test data";
  ofs.close();

  ASSERT_OK_AND_ASSIGN(auto id, GetFileCacheIdentifier(temp_file));
  // Split the ID into {timestamp}_{filesize}. We avoid using MatchesRegex
  // because gtest's simplified regex engine on Windows doesn't support
  // character classes.
  std::vector<std::string> parts = absl::StrSplit(id, '_');
  ASSERT_EQ(parts.size(), 2);

  // The first part is the last modified timestamp, which can be negative in
  // some environments.
  absl::string_view ts = parts[0];
  if (ts.starts_with('-')) {
    ts.remove_prefix(1);
  }
  EXPECT_FALSE(ts.empty());
  for (char c : ts) {
    EXPECT_TRUE(c >= '0' && c <= '9');
  }

  // The second part is the file size. "test data" is exactly 9 bytes.
  EXPECT_EQ(parts[1], "9");

  EXPECT_FALSE(GetFileCacheIdentifier("non_existent_file").ok());
}

TEST(FileUtilTest, FileExists) {
  ASSERT_OK_AND_ASSIGN(auto temp_file,
                       JoinPath(testing::TempDir(), "exists_test.txt"));
  EXPECT_FALSE(FileExists(temp_file));

  std::ofstream ofs(temp_file);
  ofs << "data";
  ofs.close();

  EXPECT_TRUE(FileExists(temp_file));
}

TEST(FileUtilTest, DeleteStaleCaches) {
  std::string temp_dir = testing::TempDir();
  std::string model_name = "test_model.tflite";
  ASSERT_OK_AND_ASSIGN(auto model_path, JoinPath(temp_dir, model_name));

  std::ofstream model_ofs(model_path);
  model_ofs << "model data";
  model_ofs.close();

  ASSERT_OK_AND_ASSIGN(auto id, GetFileCacheIdentifier(model_path));
  std::string identifier = absl::StrCat("_", id);

  ASSERT_OK_AND_ASSIGN(
      auto valid_cache,
      JoinPath(temp_dir, absl::StrCat(model_name, ".suffix", identifier)));
  std::ofstream valid_ofs(valid_cache);
  valid_ofs << "valid cache";
  valid_ofs.close();

  ASSERT_OK_AND_ASSIGN(
      auto stale_cache,
      JoinPath(temp_dir, absl::StrCat(model_name, ".suffix_stale")));
  std::ofstream stale_ofs(stale_cache);
  stale_ofs << "stale cache";
  stale_ofs.close();

  ASSERT_OK_AND_ASSIGN(auto unrelated, JoinPath(temp_dir, "unrelated.txt"));
  std::ofstream unrelated_ofs(unrelated);
  unrelated_ofs << "unrelated data";
  unrelated_ofs.close();

  EXPECT_TRUE(DeleteStaleCaches(temp_dir, model_path, ".suffix").ok());

  EXPECT_TRUE(FileExists(model_path));
  EXPECT_TRUE(FileExists(valid_cache));
  EXPECT_FALSE(FileExists(stale_cache));
  EXPECT_TRUE(FileExists(unrelated));
}

}  // namespace
}  // namespace litert::lm
