# 计算机视觉：从像素到三维重建

以 Richard Szeliski《Computer Vision: Algorithms and Applications（第 2 版）》为底本的中文视频课程：144 集，26.2 小时，从数学工具箱一路讲到三维重建与计算摄影。

> 全 144 集已制作完成，2026 年 11 月 7 日起在 B 站每日更新 5 集。
> 视频合集：[计算机视觉：从像素到三维重建｜144 集完整课](https://space.bilibili.com/54401970/channel/collectiondetail?sid=9510150)

## 这套课是什么

- **一条主线**：像素 → 图像滤波 → 特征与匹配 → 多视图几何 → SfM/SLAM → 三维重建 → 计算摄影。
- **零基础入口**：前 8 集是数学工具箱（线代、概率、卷积、采样），只要求高中数学。
- **可复现**：每集配幻灯片、口播稿、实验代码和数值结果；代码冷启动可跑。
- **不抄图**：教材插图一律重绘或程序生成，图注注明来源；教材 PDF 不随仓库分发。

## 先看这里

| 想做什么 | 看哪里 |
|---|---|
| 看 144 集完整目录 | [COURSE_MAP.md](COURSE_MAP.md) |
| 从第一集开始学 | [E000 课程宣传片](episodes/E000_course_promo/README.md) → [E001 线性代数](episodes/E001_linear_algebra_survival/README.md) |
| 直接进正课 | [E009 什么是计算机视觉](episodes/E009_what_is_computer_vision/README.md) |
| 查英文术语 | [GLOSSARY.md](GLOSSARY.md)（1028 条，含 TTS 读音） |
| 合法获取教材 | [docs/GETTING_THE_BOOK.md](docs/GETTING_THE_BOOK.md) |
| 看每集资料 | 课程地图里点集号 |

## 前 10 集

| 集号 | 标题 | 时长 |
|---|---|---|
| E000 | 人人都能学懂的计算机视觉（课程宣传片） | 3分17秒 |
| E001 | 线性代数够用就行：向量、矩阵与三种乘法理解 | 11分33秒 |
| E002 | 齐次坐标：让"平移"也变成乘法 | 9分31秒 |
| E003 | 导数、偏导与梯度的几何意义 | 8分37秒 |
| E004 | 矩阵求导与最小二乘的灵魂（一次到位） | 8分55秒 |
| E005 | 概率速通：随机变量、期望、方差、似然 | 9分37秒 |
| E006 | 贝叶斯思维：先验、似然、后验与 MAP | 9分09秒 |
| E007 | 卷积到底是什么：加权平均的数学化 | 8分49秒 |
| E008 | 图像即函数：采样、量化与混叠 | 10分35秒 |
| E009 | 什么是计算机视觉？为什么它这么难 | 12分04秒 |

完整 144 集见 [COURSE_MAP.md](COURSE_MAP.md)。

## 仓库结构

```
episodes/EXXX_slug/
  README.md              本集学习页（对应章节、时长、资料入口）
  slides/deck.html       幻灯片，浏览器直接打开
  script/script.md       口播稿
  code/                  实验代码、数值结果、程序生成图
assets/shared/           幻灯片模板、字体与 KaTeX
COURSE_MAP.md            144 集总表
GLOSSARY.md              中英术语对照（1028 条）
docs/GETTING_THE_BOOK.md 教材官方获取方式
```

## 怎么用

1. 先看视频（B 站合集），再打开本集 `slides/deck.html` 对照；
2. 想复现数字，进 `code/` 跑脚本；每集的 `code/outputs/*.txt` 是课程里引用的实际输出；
3. 术语不确定时查 `GLOSSARY.md`，全课程译名统一。

## 更新节奏

2026-11-07 起，每晚 20:00 更新 5 集，29 天更完。合集按课程顺序排列；建议从 E000 或 E009 开始。

## 版权与来源

- 课程文案、幻灯片、代码、重绘示意图：本仓库原创。
- 教材：Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. 官方免费获取：<https://szeliski.org/Book>。本仓库不包含教材 PDF 或教材原图。
- 外部论文专题会标注论文出处；论文版权归原作者与出版方。

## License

- 代码：[MIT](LICENSE)
- 文案、幻灯片、图：[CC BY-NC-SA 4.0](LICENSE-DOCS)
