#!/usr/bin/env bash
# run_by_names.sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ==== 所有配置（names 作为 key） ====
names=(
  "switzerland_seq4@8@sunny@500"
  "USA_seq5@8@sunset@300"
  "cloudy_400"
  "DJI_20250612194040_0013_V_900"
  "DJI_20250612173353_0002_V"
  "DJI_20250612174308_0001_V"
  "DJI_20250612182017_0001_V"
  "DJI_20250612182732_0001_V"
  "DJI_20250612183852_0005_V"
  "DJI_20250612193704_0010_V"
  "DJI_20250612193930_0012_V"
  "DJI_20250612194150_0014_V"
  "DJI_20250612194622_0018_V"
  "DJI_20250612194903_0021_V"
  "feicuiwan_sim_seq1"
  "feicuiwan_sim_seq2"
  "feicuiwan_sim_seq3"
)


# ==== 你想运行哪些 name？====
target_names=(
  # "DJI_20250612173353_0002_V"
  # # "DJI_20250612173353_0002_V_test"

  # "DJI_20250612182732_0001_V"
  # "DJI_20250612174308_0001_V"
  # "DJI_20250612182017_0001_V"
  # "DJI_20250612183852_0005_V"
  # "DJI_20250612194903_0021_V"  

  # "DJI_20250612193704_0010_V"
  # "DJI_20250612193930_0012_V"
  # "DJI_20250612194150_0014_V"
  # "DJI_20250612194622_0018_V"
  # "feicuiwan_sim_seq1"
  "feicuiwan_sim_seq2"
  # "feicuiwan_sim_seq3"
)

# ==== 从 txt 中读取 init_euler 和 init_trans ====
read_pose_from_file() {
  local name="$1"
  local pose_file="/mnt/sda/MapScape/query/poses/${name}.txt"

  if [[ ! -f "$pose_file" ]]; then
    echo "❌ 找不到 pose 文件: $pose_file"
    return 1
  fi

  local first_line
  first_line=$(head -n 1 "$pose_file")

  # 解析：name lon lat alt roll pitch yaw
  read -r _ lon lat alt roll pitch yaw <<< "$first_line"

  # 构造 init_euler 和 init_trans
  init_euler="[$pitch, $roll, $yaw]"
  init_trans="[$lon, $lat, $alt]"

  echo "$init_euler|$init_trans"
  return 0
}  # ✅ 这一行必须存在，否则后续 for/if 会错乱

# ==== 遍历 target_names ====
for target_name in "${target_names[@]}"; do
  index=-1
  for i in "${!names[@]}"; do
    if [[ "${names[$i]}" == "$target_name" ]]; then
      index=$i
      break
    fi
  done

  if [[ $index -ge 0 ]]; then
    result=$(read_pose_from_file "$target_name")
    if [[ $? -ne 0 ]]; then
      echo "❌ 无法读取 $target_name 的位姿，跳过"
      continue
    fi

    IFS='|' read -r euler trans <<< "$result"

    echo "==== 正在运行 $target_name ===="
    echo "euler : $euler"
    echo "trans : $trans"

    echo "--- targetloc"
    python /home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/main.py \
      --config "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/configs/feicuiwan_sim.yaml" \
      --init_euler "$euler" \
      --init_trans "$trans" \
      --name "$target_name"

    echo -e "==== 运行 $target_name 结束 ====\n"

    # ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    # ps aux | grep multiprocessing.resource_tracker | grep -v grep | awk '{print $2}' | xargs kill -9 || true
  else
    echo "❌ 未找到 name=$target_name 对应的配置"
  fi
done
