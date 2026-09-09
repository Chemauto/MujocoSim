# assets/ 资产库

## 目录结构

```
assets/
└── robots/          # 机器人库:一个文件夹 = 一种机器人
    ├── go2/         # go2.xml + assets/*.obj + robot.yaml
    └── g1/          # g1_29dof.xml + meshes/*.STL + robot.yaml
```

## 添加新机器人

1. 新建文件夹 `assets/robots/<机器人名>/`,放入 MuJoCo MJCF 机器人模型
   (`<mujoco>` 根元素、worldbody 里只有一个根 body,推荐从
   [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) 获取)。
2. 在文件夹里写一个 `robot.yaml`:

```yaml
name: mybot
xml: mybot.xml                    # MJCF 文件名(相对本文件夹)
base_body: base_link              # 根 body 名(带 freejoint;没有 freejoint 仿真器会自动补)
default_spawn: [0, 0, 0.5]        # 默认出生位置,可被场景配置的 spawn 覆盖
view_extent: 1.0                  # viewer 初始视野范围
description: "一句话描述"
```

3. 在 `configs/` 里复制一份 yaml,把 `robot:` 改成 `<机器人名>` 即可换机器人,
   代码零改动。

注意:机器人 MJCF 里 worldbody 层的 floor/light 会在组装时被自动剥离
(场景由 `simulation/builder.py` 统一提供),body 内部的相机等会保留。
