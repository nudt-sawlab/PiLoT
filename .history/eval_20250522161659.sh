#!/usr/bin/env bash
# run_all.sh

# 把要跑的 config 列表写在数组里
configs=(
  "configs/switzerland_seq4@8@foggy@500@VGG@LM20.yaml"
    "configs/switzerland_seq4@8@foggy@500@Unet_fusion@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@500@VGG@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@500@Unet_fusion@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@400@VGG@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@400@Unet_fusion@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@300@VGG@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@300@Unet_fusion@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@200@VGG@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@200@Unet_fusion@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@100@VGG@LM20.yaml"
    # "configs/switzerland_seq7@8@cloudy@100@Unet_fusion@LM20.yaml"
)

for cfg in "${configs[@]}"; do
  echo "==== 开始运行 $cfg ===="
  python eval.py --config "$cfg"
  echo "==== 完成运行 $cfg ====\n"
done
