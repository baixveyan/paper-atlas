# 论文馆 · 研读 / Paper Atlas

从问题出发，把顶会论文拆解成方法、公式、实验与证据边界，生成可持续更新的中文解析与讲解网站。

> 在线站点：<https://baixveyan.github.io/paper-atlas/>

## 当前进度

| 指标 | 数值 |
| --- | --- |
| 官方主会论文清单 | 4,042 篇（CVPR 2026） |
| 已完成原文解析与讲解 | 106 篇 |
| 正文与配图联合返工 | 98 / 106 进行中 |

## 目录结构

```
paper-atlas/
├── papers/                 # 论文解析内容（Markdown）
│   └── CVPR2026/           #   会议清单、进度、主题目录
├── site/                   # 静态站构建器（build.py / check.py）
├── maintenance/            # 数据管线与维护脚本
│   ├── build_paper_index.py   # 生成解析索引与站点数据
│   ├── cvpr_inventory.py      # 拉取官方开放获取清单
│   └── paper-sources/         # PDF 源文件缓存（本地，gitignore）
└── .github/workflows/
    └── paper-pages.yml        # GitHub Pages 自动构建部署
```

## 本地构建

```bash
python maintenance/build_paper_index.py --check   # 校验解析结构与索引
python site/build.py                              # 生成静态站 → site/dist
python site/check.py                              # 站点检查与修订一致性
```

仅依赖 Python 标准库 + vendored Markdown（`maintenance/site-deps/`），无外部安装要求。

## 一篇解析的结构

每篇论文按 10 节组织：

1. 一句话总结
2. 研究背景与动机
3. 方法总览
4. 核心创新与关键设计
5. 实验与关键结果
6. 消融实验与证据强度
7. 局限与讨论
8. 复现提示
9. 相关工作与阅读顺序
10. 来源与校验

## 部署

push 到 `main` 分支且涉及 `papers/**`、`site/**`、`maintenance/build_paper_index.py` 时，GitHub Actions 自动构建并部署 GitHub Pages（见 `.github/workflows/paper-pages.yml`）。

## 许可

解析内容与构建器均随本仓库发布；论文原文版权归原作者所有，本站仅提供中文解析与讲解，每篇均附官方来源链接。
