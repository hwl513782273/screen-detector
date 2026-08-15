# -*- coding: utf-8 -*-
"""
框选屏幕检测工具 —— PySide6 图形界面

功能：
  - 手动"框选监控区域"（唯一监控区域来源；已移除「抓取应用窗口」来源）
  - 多模板管理（增 / 删 / 改名 / 启停 / 预览 / 重置学习），非一键清空
  - 检测状态机：图案出现 -> 响铃；消失 -> 自动停铃；手动『暂停响铃/命中/误报』-> 暂停响铃；可在「检测状态」区切换响铃恢复方式（手动恢复 / 下次新命中自动恢复 / 自定义秒数后自动恢复）
  - 命中反馈（命中 / 误报）驱动自学习，动态调整阈值并精炼模板
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QImage, QFont, QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QPushButton, QLabel, QComboBox, QListWidget, QListWidgetItem,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QFileDialog,
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QFrame, QSizePolicy,
)

from detector_core import Detector, DEFAULT_SCALES

# 文字检测去抖参数：OCR 并非每帧都稳定，需连续若干帧命中才确认、连续若干帧缺失才清除，
# 避免"测试能命中、实际监控却闪烁失灵"。
TEXT_HIT_CONFIRM = 1    # 1 帧命中即确认，与「测试文字检测」行为一致（单帧有字即响铃）
TEXT_CLEAR_CONFIRM = 3  # 连续 3 帧未命中才清除（抗偶发漏识别，避免响铃闪烁）
# OCR 不再单独节流：每次检测循环（即『检测间隔』）在主线程跑一次 OCR，保证"按检测间隔持续文字检测"。
from capture_utils import (
    get_manager,
    screen_capture_access_status,
    request_screen_capture_access,
)
from config_store import ConfigStore
from ocr_utils import recognize_text, text_matches, vision_available, _normalize_text


# ---------------------------------------------------------------------------
# 更新日志（"关于"弹窗展示）
# ---------------------------------------------------------------------------
CHANGELOG = [
    ("v6.48", "修复检测图案详情区截断与整体布局", [
        "保持窗口 880×963 不变。",
        "将右栏『检测图案』详情区从水平布局改为垂直布局：预览图在上、表单在下，彻底解决名称/阈值/匹配度/提示音等文字和按钮的截断、重叠问题。",
        "压缩『默认提示音』和『检测节奏/参数』分组高度，把空间让给详情区；多尺度匹配勾选与间隔/冷却放到同一行。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、框选即预览、命中跟随等）。",
    ]),
    ("v6.47", "修复监控预览以中心为锚点缩放", [
        "保持窗口 880×890 不变。",
        "修复 v6.46 缩小时画面偏向左上角的问题：改为以画面中心对齐预览区中心为基准，缩小时从中心向四周均匀收缩，四周同时出现黑边。",
        "平移操作仍相对于中心位置生效，重置按钮恢复默认铺满视图。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音等）。",
    ]),
    ("v6.46", "文字提示音改为单行滚动 + 监控预览允许无限缩小", [
        "保持窗口 880×890 不变。",
        "右栏『各文字提示音』列表改为每次只完整显示一行，超出项通过垂直滚动条查看，避免开始检测后文字被遮挡。",
        "监控预览缩放放宽：点击『－』可无限缩小，缩到比预览区小后周围以黑边补齐，便于观察整体或极小区域。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音等）。",
    ]),
    ("v6.45", "监控预览加大到 432×160 并修复文字提示音列表截断", [
        "保持窗口 880×890 不变。",
        "监控预览区域从 240×160 横向加大到 432×160，框选画面显示更长更清晰。",
        "右栏『各文字提示音』列表高度 100→150，避免开始检测后目标文字行被截断。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音等）。",
    ]),
    ("v6.44", "修复宽扁监控区域预览显示过小问题", [
        "保持窗口 880×890 不变。",
        "监控预览区域从 200×128 加大到 240×160，画面更长更高。",
        "预览默认改为 cover 铺满模式（无黑边），宽扁区域（如 246×72）能占满整个 preview。",
        "保留 －/＋/方向键/重置 按钮，用户可缩放平移查看边缘细节。",
        "检测图案列表高度 88→60、详情区最小高度 130→90，为 preview 腾出空间。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音等）。",
    ]),
    ("v6.43", "修复预览显示不全并增加缩放平移控制", [
        "保持窗口 880×890 不变。",
        "修复『各文字提示音』列表中文字被截断/遮挡：固定每行 widget 与 item 高度为 26px，文字标签和下拉框高度固定 20px 并垂直居中。",
        "修复监控预览画面显示不全：改为默认按 QLabel 尺寸等比完整显示框选区域，宽扁区域不再被居中裁剪。",
        "在监控预览下方增加紧凑的缩放/平移按钮组（＋/－/←/↑/↓/→/重置），默认完整显示，可放大查看细节并平移，不影响整体布局。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.42", "修复各文字提示音列表右侧贴边/截断", [
        "保持窗口 880×890 不变。",
        "右栏『各文字提示音』列表关闭水平滚动条，并根据列表可视区宽度固定每行 item widget 宽度，避免下拉框右侧贴边或被裁切。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.41", "增高监控预览并压缩检测图案下方空白", [
        "保持窗口 880×890 不变。",
        "左侧『监控预览』区域从 140×90 增大到 200×128，预览画面更大更清晰。",
        "左侧『检测图案』列表下方空白压缩：列表高度 140→110，多选提示文字固定 18px 不再被拉伸，按钮紧贴列表下方。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.40", "优化三处布局（保持 880×890）", [
        "保持窗口 880×890 不变。",
        "左侧『检测图案』列表高度 100→140，减少列表下方大片空白。",
        "右侧『各文字提示音』列表高度 70→100，并为每行 item 设置 sizeHint，解决下拉框文字显示不全/截断问题。",
        "右侧『学习反馈记录』内容栏取消最大高度 56，改为最小 120 + Expanding，占满整个分组框。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.39", "固定 880×890 并修复分组框标题消失", [
        "窗口固定为 880×890（用户拖动 v6.38 后确认的最佳高度）。",
        "修复 QGroupBox 标题在真实运行中消失的问题：margin-top 26→20px，title top 偏移 -15→-8px，使分组框标题完整显示。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.38", "解除高度锁定，方便用户自定义窗口高度", [
        "窗口宽度保持 880 锁定，高度解除固定：改为 setFixedWidth(880) + setMinimumSize(880, 650) + resize(880, 780)。",
        "用户可自由拖动窗口下边缘调整高度，找到右栏标签完整显示且整体舒适的尺寸后，把高度数值反馈给开发即可生成最终固定高度版。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.37", "修复右栏标签被压扁（所见即所得布局）", [
        "修复 880×780 紧凑布局下右栏『匹配 / 目标 / 文字匹配 / 各文字提示音』等说明标签被压缩成 2-3px 消失的问题：收紧 QGroupBox 内边距(20/12→14/10)与顶部外边距(26→22)、右栏分组间距(10→6)，释放纵向空间使标签完整显示。",
        "交付前改用真实代码离线渲染截图核对布局，确保预览图与打包后实际界面完全一致（所见即所得）。",
        "功能逻辑不变（暂停/启动响铃、匹配模式、目标文字、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.36", "依据参考图重排为 880×780 紧凑双栏", [
        "窗口高度 980→780：去掉底部空白与分组间冗余间距，整体更紧凑。",
        "『学习反馈记录』从底部左栏移回右栏底部，左栏只保留监控区域 / 监控预览 / 检测图案三段，消除大片空白。",
        "收紧检测图案列表(110→100)、图案详情区(170→150)、各文字提示音列表(90→70)高度，确保 880×780 内完整显示、无滚动条。",
        "保留 v6.27.4 以来全部功能（暂停/启动响铃、匹配模式持久化、目标文字持久化、参与检测勾选、各文字独立提示音、学习反馈记录等）。",
    ]),
    ("v6.35", "回退布局防遮挡（标题完整 + 右栏不压扁）", [
        "保持窗口固定 880×980 不变，修复分组标题被截断与右栏文字提示音输入框被压扁：加大 QGroupBox 顶部边距与标题偏移，确保所有分组标题完整显示不再被裁。",
        "把『学习反馈记录』从右栏移到底部左栏，利用 980 高度给日志独立空间；逐文字提示音下拉框设最小宽度 100、长标签缩短，右栏不再出现挤压/水平滚动条。",
    ]),
    ("v6.34", "回退到 v6.27.4 稳定布局并修复显示问题", [
        "按用户要求将 v6.28 起的实验性新 UI 回退为 v6.27.4 稳定布局：窗口固定 880×980，主界面恢复为左/右双栏（无滚动区），分组标题完整显示、不再被截断。",
        "修复 v6.32/v6.33 中监控预览、检测图案详情、文字提示音输入框被挤压/截断的问题：所有控件在固定窗口内完整铺开，无水平滚动条。",
        "保留 v6.27.4 以来的全部功能：暂停/启动响铃、匹配模式持久化、目标文字持久化、图案参与检测勾选、各文字独立提示音、学习反馈记录等。",
    ]),
    ("v6.33", "修复 900×1046 双栏截断/挤压问题", [
        "修复分组标题被裁剪：QGroupBox 顶部边距 16→22px、标题偏移 -9→-12px、内边距调整，确保『监控区域/监控预览/检测图案/检测模式/检测节奏/默认提示音/检测状态/学习反馈记录』等标题完整显示不被截断。",
        "修复右栏过窄导致文字提示音输入框被压扁：双栏比例由 5:4 改为 4:5，右栏获得更多水平空间；左右滚动区均禁用水平滚动条，强制内容自适应列宽。",
        "修复长标签撑出滚动条：『各文字提示音』标签开启自动换行；『测试文字检测』按钮简化为『测试』并保留 tooltip；逐文字提示音列表项文字标签最小宽度 80→50。",
        "重构图案详情区：预览图由左侧改到上方居中，表单字段独占整栏宽度，避免 900 宽下左右挤压导致按钮/标签重叠。",
        "移除重复定义的旧 GLOBAL_QSS 块，只保留一份设计系统，避免样式意外叠加。",
        "窗口仍默认 900×1046，所有功能逻辑（关闭记忆、暂停/启动响铃、反馈卡、检测模式等）保持不变。",
    ]),
    ("v6.32", "窗口默认 900×1046 · 全部内容清晰铺开", [
        "按用户指定将窗口默认尺寸定为 900×1046（最小 900×720，可缩放但默认即此尺寸），比之前 980×720 更高，竖向空间充裕。",
        "放大关键区域的最小高度：监控预览井 220→300、图案详情区 150→200、图案列表最大高度 130→170、各文字提示音列表 90→110，使所有控件在 900×1046 下都舒展、无挤压、无截断，全部功能与交互逻辑原样保留。",
    ]),
    ("v6.31", "UI 改为双栏紧凑布局", [
        "按用户反馈将 v6.28 的浅色三栏改为『双栏紧凑』方案：左栏为监控区域 + 监控预览井 + 检测图案（列表/勾选/详情），右栏为检测模式/文字 + 检测节奏/参数 + 默认提示音 + 检测状态 + 学习反馈记录，底部常驻开始/暂停响铃控制栏。",
        "窗口由默认 1180×780 收到 980×720（最小 900×640），卡片内边距、字号、按钮高度同步压缩，监控预览最小高度降到 220，整体不再撑满小屏；所有功能与交互逻辑完全保留。",
    ]),
    ("v6.28", "全新 UI（浅色三栏 + 监控井 + 底栏控制）", [
        "按 UI 设计方案重写界面布局：浅色 macOS 风格，左侧『配置』（监控区域/检测图案/提示音）、中间『监控井』（大预览 + 命中确认悬浮卡 + 检测状态 + 学习反馈记录）、右侧『精确调节』（检测模式/目标文字/各文字提示音/检测节奏参数），底部常驻控制栏（开始/停止检测 + 暂停/启动响铃 + 状态）。",
        "新增全局 QSS 设计系统（系统蓝 #0071e3、语义状态色、原生质感按钮/输入框/分组框），所有功能与交互逻辑（关闭记忆、暂停/启动响铃、参与检测勾选、反馈卡、测试匹配）完全保留，仅重做视觉与信息架构。",
        "窗口改为可缩放（最小 960×680，默认 1180×780），左/右栏内容可滚动，避免小屏下控件被裁切。",
    ]),
    ("v6.27.4", "关闭记忆检测配置 + 暂停/启动响铃标签", [
        "自动记住每次关闭时的检测配置：目标文字、各图案的『参与检测』勾选状态、以及匹配模式（仅图案/仅文字/文字优先·图案兜底）均随配置落盘，重启后自动恢复，无需重新设置。",
        "将响铃控制按钮文案由『停止响铃/恢复响铃』改为『暂停响铃/启动响铃』，形成清晰开关：响铃中或待命时显示『⏸ 暂停响铃』（点击即静音保持监控），暂停后显示『▶ 启动响铃』（点击即恢复），idle 态点击也可切换。",
        "匹配模式此前未被持久化，每次启动都会重置为默认『仅文字』；现已与模板、目标文字一并保存与恢复。",
    ]),
    ("v6.27.3", "响铃改为『暂停响铃 + 手动恢复』", [
        "修复『响铃后点击停止响铃/命中/误报会彻底停止响铃』的问题：原 muted_this_hit 仅在目标完全消失后才解除静音，持续目标下永不再响。",
        "改为『暂停响铃』模型：点击『停止响铃』/『命中』/『误报』后仅静音并保持监控（暂停），按钮变为蓝色『▶ 恢复响铃』；点击『恢复响铃』才解除暂停，后续命中重新响铃。",
        "暂停状态在目标消失/重新出现期间保持不变，全程等待用户手动恢复，不再自动解除；停止检测时彻底重置暂停状态。",
    ]),
    ("v6.27.2", "修复图案详情区显示截断", [
        "修复图案详情区（当前最佳匹配度、保存图案图片按钮）在固定窗口 880×980 下与上方行重叠/截断的问题。",
        "将『测试匹配』与『保存图案图片』合并到同一行，缩小预览图与行间距，并调高详情区最小高度、压低模板/文字声列表高度，保证所有控件完整可见且窗口尺寸不变。",
    ]),
    ("v6.27.1", "修复图案详情区显示截断", [
        "修复图案详情区（名称、基准阈值、有效阈值、当前最佳匹配度、保存图案图片、图案提示音）在固定窗口 880×980 下被模板列表挤压导致显示不全/重叠的问题。",
        "限制模板列表最大高度并强制详情区最小高度 160px，保证详情区所有控件完整可见；保持整体窗口固定尺寸不变。",
    ]),
    ("v6.27", "每个文字独立选声·停止响铃持续到下次命中", [
        "每个文字可独立选声：文字检测区由『整组共用一个提示音』改为逐文字列表，每行一个文字 + 独立提示音下拉；填多个文字（逗号/、隔开）时各自可指定不同声音，命中时按该文字的声音响铃，未指定则回退最下方全局默认。映射随配置持久化。",
        "声明默认提示音：当所有图案与文字都未单独指定提示音时，一律以最下方『默认提示音』（全局声音）响铃；任一图案/文字单独指定后以其为准。已在全局声音下拉处标注与提示。",
        "停止响铃持续到下次命中：手动点『停止响铃』后，本次命中的目标消失前不再响铃（移除原先『静音期到期自动恢复』逻辑），目标消失后才解除静音，下次命中重新开始——避免静态目标下反复被打扰。",
        "命中确认也关闭响铃：图案命中后弹出的反馈卡，无论用户点『命中』还是『误报』，确认后都立即停止本次响铃（此前仅『误报』会停铃），确认即静音，直到下次命中。",
    ]),
    ("v6.26", "图案/文字自动保存·不同图案文字不同提示音", [
        "目标文字自动保存：之前仅图案模板被持久化，『目标文字』与『文字匹配方式』重启即丢失。现已在配置中一并落盘，并在输入框/下拉变化时即时写入（输入即存），开机无需重新输入。",
        "不同图案/文字可配置不同提示音：每个图案在详情区新增『图案提示音』下拉（默认=全局）；文字检测区新增『文字提示音』下拉（默认=全局）。命中时按触发来源播放对应声音——图案命中播该图案的声音，文字命中播文字目标的声音，互不影响。",
        "提示音配置持久化：图案声音写入模板存档、文字声音写入全局配置，跨会话保留；响铃状态机改为按来源解析声音，复用 v6.25 『先停上一个再播新的』防重叠逻辑。",
    ]),
    ("v6.25", "修复文字检测弹窗双显示·修复提示音重叠", [
        "修复『检测文字检验会出现两次窗口』：根因是稳健弹窗 _show_msg 在 macOS 上先 box.show() 创建并显示原生窗口、紧接着又 box.exec() 再次显示，导致同一个提示框出现两个窗口；现已去掉多余的 show()，只保留 exec() 单次模态显示（与『关于』弹窗一致），文字检测测试/各类提示均只显示一个窗口。",
        "修复『命中后提示音重叠』：根因是 SoundPlayer 每次播放都新建 NSSound 却从不保存引用，旧声音还在播就被新的叠加，且 stop() 只遍历从不填充的缓存等于空操作。现已保存『当前播放实例』，每次播放新声音前先停掉上一个，stop() 真正停止当前声音，连续响铃不再叠加。",
    ]),
    ("v6.24", "CPU/内存占用优化·临时文件自动清理", [
        "整体 CPU 占用优化：苹果原生 OCR（Vision）在识别前先把画面按最长边 ≤ 1280px 自动等比缩小再识别，识别坐标按比例映射回原图，文字检测精度不变，但 OCR 计算量随像素面积下降（最长边减半时约降 75%），检测节拍显著变轻。",
        "预览节拍降频：监控画面刷新由 100ms(10fps) 调整为 200ms(5fps)，对监控足够流畅且抓屏/缩放/绘制开销减半；窗口最小化或不可见时暂停预览抓屏，彻底避免后台空转占 CPU。",
        "内存占用优化：OCR 内部 RGBA 缓冲与 CFData 随降采样同步缩小；沿用 NSAutoreleasePool 及时释放 Vision 临时对象，长时间运行内存平稳不再累积。",
        "临时文件自动清理（防止存储膨胀）：① 启动时扫描配置目录，自动删除『模板已从列表删除但仍残留的孤立图片 PNG』与上次异常遗留的 *.tmp 文件；② 打包脚本 build.sh 每次构建成功后自动清理 build/、dist/ 与 /tmp/old_build_* 等大体积中间产物（仅交付的版本化 DMG 保留，遵循只增不删）。",
    ]),
    ("v6.23", "修复长时间运行闪退·图案勾选参与检测", [
        "修复『运行一段时间之后闪退』：根因是 Apple Vision OCR 每次识别都会创建 CGImage / VNImageRequestHandler / VNRecognizeTextRequest 等 Obj-C 对象，之前未显式 drain 自动释放池，长时间持续识别让其无限累积直至被 macOS 内存压力杀掉。现已把 recognize_text 整段包进 NSAutoreleasePool，每帧 OCR 结束即释放全部临时 Obj-C 对象。",
        "图案检测改为『勾选参与检测』：模板列表每个图案前加勾选框——勾选才参与本轮检测；取消勾选则跳过该图案但仍保留模板（相关配置持久化）。",
        "选中的多个图案之间为『或』关系：只要勾选的任意一个图案命中即触发响铃（满足『选两个，中其中一个便响铃』）。",
    ]),
    ("v6.22", "检测间隔改为秒·去变化检测·命中来源·修复花屏·固定窗体·自定义提示音", [
        "检测间隔单位由毫秒改为秒：UI 标签、默认值、保存键均改为秒，定时器启动时自动乘以 1000，预览节拍保持高频。",
        "去除『变化检测模式（实时高亮变化区域）』复选框及对应状态/绘制逻辑，界面更简洁。",
        "命中时明确显示命中来源：『图案命中』、『文字命中』或『文字+图案』，最近命中详情同步标注。",
        "修复图案命中后点『命中』反馈导致模板缩略图花屏：numpy_to_pixmap 现在会先把 float32 精炼图转 uint8、确保 contiguous，再转 QPixmap。",
        "主窗口固定为 880×980，禁止缩放，确保所有功能分区完整显示。",
        "新增自定义提示音：提示音下拉框增加『自定义…』项，可浏览选择本地音频文件（aiff/wav/caf/mp3），配置持久化。",
    ]),
    ("v6.21", "遮挡自恢复·检测永不中断·学习反馈持续", [
        "修复『有物理遮挡后整个检测停止、连学习反馈记录都没了』：根因是遮挡场景下某帧匹配/OCR/绘制抛出的未捕获异常会冒泡到主线程 QTimer 槽，使检测节拍静默中断（之前只在后台线程加过容错，主线程这层没有）。",
        "检测节拍 `_detect_step` 与预览节拍 `_preview_step` 整段包 try/except：任何单帧异常只跳过本帧并打印，监控持续运行，绝不停止、不清空状态、不停定时器。",
        "遮挡自恢复：抓取暂不可用（窗口遮挡/全屏切换/权限问题）时只跳过本帧、持续重试；状态栏显示『⚠ 区域被遮挡，监控持续重试中（已 Xs）』，并在『学习反馈记录』里记录遮挡起止（遮挡开始 / 解除恢复），让监控与反馈面板始终显示「仍在运行」，不再像「死了」。",
        "遮挡期间不记录任何假负样本，避免污染自学习阈值；遮挡解除后检测与命中反馈立即自动恢复。",
    ]),
    ("v6.20", "修复『窗口遮挡后检测失效』（屏幕录制权限）", [
        "根因：macOS 未授予『屏幕录制』权限时，截图里其它 App 的窗口会被系统涂成纯色/空白；监控区域一旦被别的窗口遮挡，那块画面就『消失』，模板永远匹配不上，表现为『有窗口遮挡之后检测失效』。",
        "启动时主动申请『屏幕录制』权限（弹系统授权框）；未授权时给出明确、持续的警告，并解释遮挡即失效的成因与『完全遮挡属物理限制、无法检测』。",
        "开始检测前再次校验权限：未授权则提示去『系统设置 → 隐私与安全性 → 屏幕录制』开启本 App，授权后遮挡区域即可正常抓取、检测恢复。",
        "抓取失败时状态栏明确提示『可能未授予屏幕录制权限』；监控不因单帧抓取失败而中断。",
    ]),
    ("v6.19", "持续监控（按节拍检测 + 可靠响铃）", [
        "重构检测节拍：『开始检测』后用两个独立定时器驱动——检测节拍按『检测间隔』每间隔跑一次『测试文字检测』式 OCR + 图案匹配并响铃（持续监控，不再只在点击开始那一刻响一次）；预览节拍独立高频刷新监控画面，保证实时流畅。",
        "修复『响铃只响一次/不持续』：提示音改为每次播放都新建 NSSound 实例从文件载入，规避 NSSound 共享实例连续 play 偶发不发声的问题，目标持续存在时按『响铃冷却』稳定重复响铃。",
        "移除后台工作线程（曾因 Vision OCR 在部分 macOS 环境静默停摆导致预览卡死/监控不稳），改由主线程定时器驱动，逻辑更直观、更易维护。",
    ]),
    ("v6.18", "图案列表支持多选（默认单选·批量操作）", [
        "修复『图案列表不能多选』且避免『默认就多选』：将检测图案列表选择模式改为 ExtendedSelection——默认仍是单选（编辑详情照常作用于单个图案），但可按住 ⌘/Ctrl 或 Shift 多选。",
        "『删除选中』与『重置学习』支持多选批量操作：选中多个图案时一次性删除/重置全部（记录日志含数量）；列表下方新增一行提示文字说明多选方式。",
    ]),
    ("v6.17", "文字优先检测 / 按间隔持续 OCR / 文字+图案改为兜底", [
        "默认匹配模式改为『仅文字』：直接点『开始检测』即优先文字检测（OCR）。",
        "OCR 改为每当次检测循环（即『检测间隔』）在主线程跑一次，实现『按检测间隔时间持续 OCR 文字检测』；去除原先独立的 0.25s 节流。",
        "『文字和图案』更名为『文字优先·图案兜底』并改语义为 OR：文字优先检测，文字检测不到目标时再用图案命中兜底（文字或图案任一命中即响铃）；状态栏与响铃会标明本次由『文字』还是『图案兜底』触发。",
        "监控预览本就每帧实时刷新（v6.15 已修 OCR 移主线程），本轮保持不变；仅文字模式不画模板框、其余模式照常画。",
    ]),
    ("v6.16", "移除监控来源（仅保留框选）", [
        "移除『监控来源』中的应用抓取功能（应用下拉 / 刷新应用 / 抓取该应用窗口），彻底删除对应逻辑与 `_refresh_apps` / `_grab_app_window` 方法。监控区域现在只由『① 框选监控区域』一个按钮设置，避免『抓取应用到窗口一直失效/卡死』反复困扰。左侧分组框由『监控来源』更名为『监控区域』。",
    ]),
    ("v6.15", "修复监控预览卡死（OCR 移回主线程）", [
        "修复『监控画面预览只在点文字检测时才变化、平时卡住不动』：根因是 v6.13 把 Vision OCR 移到后台工作线程，在部分 macOS/pyobjc 环境下后台线程跑 OCR 会让工作线程静默停摆（不再发 tick），预览因此冻结；而『测试文字检测』是主线程跑 OCR 所以正常。现把 OCR 移回主线程（节流 0.25s 执行），工作线程只负责抓帧+图案匹配——预览每帧稳定刷新，文字检测仍可靠；工作线程主循环再用 try/except 兜底，任何异常都不会再静默退出监控。",
    ]),
    ("v6.14", "修复抓取窗口 bounds / 文字监控兜底", [
        "修复『抓取应用窗口时报错 __NSDictioaryI has no attribute origin』：macOS 的 kCGWindowBounds 在某些系统/pyobjc 下返回 NSDictionary 而非 CGRect，现统一用 `_bounds_to_rect()` 解析（兼容 .origin/.size 结构与 X/Y/Width/Height 字典键），窗口抓取不再崩溃。",
        "修复『测试文字检测成功、但开始检测又失效』：文字命中确认改为 1 帧即触发（与测试行为一致），清除保持 3 帧未命中才清除；同时增加主线程 OCR 兜底——当工作线程 OCR 返回空且当前帧有效时，以 1 秒间隔在主线程再识别一次，避免某些环境下后台线程 OCR 不稳定导致实际监控漏字。",
    ]),
    ("v6.13", "抓取卡死修复 / 三态匹配模式 / 文字监控稳定 / 保存图案", [
        "修复『点击抓取该应用窗口一直提示正在抓取应用窗口』：抓取前先置可见状态并禁用按钮防止重复点击；全程 try/except，任何异常（如窗口查询/抓取抛错）都会复位状态栏并弹出错误提示，不再卡在『正在抓取』无法恢复；若监控中抓取还会实时把新区域同步给工作线程。",
        "新增『匹配模式』三态选择（替换原『启用文字检测』勾选框）：仅图案 / 仅文字 / 文字和图案。三者为独立模式——『文字和图案』需图案与文字同时命中才响铃，便于组合条件检测。开始检测会按模式校验（仅图案需模板、仅文字需目标文字、组合需两者皆有）。",
        "修复『测试文字检测成功、但实际监控不恒工（不稳定）』：① 把 OCR 从主线程移到工作线程，避免每帧 OCR 阻塞 UI 且保证每帧只 OCR 一次（不再因主线程排队漏帧）；② 文字命中加入去抖（连续 2 帧命中才确认、连续 5 帧未命中才清除），消除 OCR 偶发漏识别导致的响铃闪烁/失灵；③ 监控中实时把工作线程识别到的文字传入主线程匹配。",
        "新增『保存图案图片』：模板详情区新增按钮，可把当前选中模板（框选/载入的原始图案）一键导出为 PNG/JPEG 保存到任意位置（默认存到桌面），便于复用、归档或分享；在此之前模板图片仅在工具内部持久化、用户无法另存。",
    ]),
    ("v6.12", "实时监控静默崩溃修复", [
        "修复『画面里明明有目标、测试匹配能命中 0.95+，但实时监控从不提示命中/不响铃』：根因是后台工作线程与『测试匹配/反馈精炼』并发读写模板的缩放缓存与精炼图，导致工作线程抛出 KeyError 后被静默终止（线程死掉后再无 tick，界面卡在『监控中』）。现已统一加锁并让工作线程对单帧抓取/匹配异常容错（跳过本帧继续监控）。",
        "修复命中诊断键名错误：最接近匹配得分此前取错字段永远为 None，导致『未命中但接近阈值』时看不到真实得分；现已正确显示『最接近：名称 得分 / 阈值』，便于排查。",
    ]),
    ("v6.11", "文字命中 / 抓取弹窗再修复", [
        "修复『文字检测测试命中但监控不提示命中/不响铃』：Apple Vision OCR 对中文有时会在字符间插入空格/零宽字符，导致『包含任一』子串匹配漏配；现对识别文字与目标文字均做归一化（全角转半角、移除空格与零宽字符）后再匹配，预览命中框也按归一化结果标橙。",
        "修复『抓取该应用窗口点击无反应』再发：弹窗改用独立顶层窗口 + WindowStaysOnTopHint + 多次 processEvents/raise_/activateWindow，避免 macOS 上 QMessageBox 仍偶发不前置/不显示的问题。",
        "文字检测未命中时，每 5 秒向日志输出一次识别到的文字样本，便于用户排查目标与 OCR 结果差异。",
    ]),
    ("v6.10", "预览修复 / 抓取提示 / 文字检测", [
        "修复『开始检测后监控预览全黑』：后台工作线程复用主线程创建的 mss 实例在 macOS 上抓到黑帧；改为每个线程独立的 mss 实例；预览整帧接近全黑时叠加『[抓取为黑屏]』诊断字样。",
        "修复『抓取该应用窗口点击无反应』：抓取成功/失败/未选应用等所有提示统一改用稳健弹窗（raise_/activateWindow/exec），避免 macOS 上 QMessageBox 偶发不显示导致看似无反应；并同步状态栏反馈。",
        "新增『文字检测』（macOS 原生 Vision OCR）：左侧新增文字检测分组——勾选启用、输入目标文字（逗号/、隔开）、选择匹配方式（包含任一/全部包含/完全相等）、可『测试文字检测』；监控时实时 OCR 并在预览用绿/橙框标出识别文字，命中目标即响铃（与图案检测互斥）。",
    ]),
    ("v6.9", "响铃复响与提示修复", [
        "修复『识别到一个后点击停止响铃就不能再次识别』：手动停铃后增加定时自动解除静音（≥响铃冷却且不少于3秒），静态目标持续存在时也能再次响铃。",
        "修复『抓取该应用窗口成功后提示框失效』：改用显式 QMessageBox 并强制 raise_/activateWindow，避免 macOS 上弹窗不前置/不显示。",
    ]),
    ("v6.8", "响铃修复与变化检测", [
        "修复『误报』后仍继续响铃：标记误报时立即停铃，并在本次命中持续期内静音，直到图案消失再出现才恢复。",
        "修复『停止响铃』不可点击：v6.7 改为仅响铃中可点属回归，现检测进行中按钮始终可点（响铃/未响铃均可），停止检测时才禁用。",
        "修复『抓取该应用窗口』成功无任何提醒：抓取成功后弹出『抓取成功』提示框，明确告知已设为监控区域与应用名。",
        "新增『变化检测模式』：实时监控预览可勾选该模式，实时高亮相对上一帧（当前抓取）发生变化的位置，便于观察动态变化。",
    ]),
    ("v6.7", "状态与交互修复", [
        "修复『停止检测』后检测状态卡在『已停止』：停止时干净重置响铃/静音状态并置否检测标志，重新开始立即切回『监控中』，并丢弃停止后的迟到 tick。",
        "修复基准阈值无法选到 0.2：滑块下限由 0.50 放宽到 0.20，低相似度目标也能手动调到更低阈值命中。",
        "修复点击『停止响铃』后按钮无变化：点击后立即变为『已停止响铃』灰色禁用态，再次出现目标时自动恢复为可点的『停止响铃』。",
    ]),
    ("v6.6", "四项问题修复", [
        "修复『框选监控区域』选区不可见：重写覆盖层绘制（避开 QRegion 裁剪），选中区外框为醒目红实线+红虚线，选区内露出真实画面。",
        "修复『抓取该应用窗口』无效：增加 pid 守卫、结果校验、抓取验证与最前台窗口兜底，并给出清晰错误提示。",
        "修复『从图片载入』无法命中：载入时按当前监控画面自动校准阈值（分辨率/渲染差异也能命中），放宽多尺度至 0.5x~1.5x，默认阈值降至 0.75。",
        "修复『停止响铃』无反应：点击后立即消音（杀掉尾音），检测进行中按钮始终可点，并持久显示『已停止响铃』状态。",
    ]),
    ("v6.5", "交互与响铃优化", [
        "框选监控区域时，选中区域外框改为红色虚线，更醒目。",
        "命中/误报反馈从独立置顶弹窗改为嵌入主窗口，不再遮挡其它内容。",
        "修复『停止响铃』无效：手动停止后，在本次命中持续期内保持静音，直到图案消失并再次出现才恢复响铃。",
        "默认阈值降至 0.80，多尺度搜索进一步扩大，任意位置出现目标均更易命中。",
    ]),
    ("v6.4", "命中增强与诊断", [
        "扩大多尺度搜索范围并启用绝对相关系数，对反色/悬停态 UI 更鲁棒，降低漏检。",
        "未命中时显示最接近模板的最佳得分与阈值，帮助判断是阈值问题还是模板不匹配。",
        "模板详情新增『测试匹配』按钮，可立即验证当前模板在当前监控区域能否命中。",
    ]),
    ("v6.3", "问题修复", [
        "修复检测图案详情区布局重叠/位置混乱。",
        "修复点击『开始检测』后界面卡死：由工作线程把画面传给 UI，避免主线程重复抓取阻塞。",
        "修复框选覆盖层黑屏/白框：先截取屏幕作为背景，选区外叠加半透明遮罩，选区内显示真实画面。",
    ]),
    ("v6.2", "本次更新", [
        "框选遮罩改为半透明（非全黑），选区内露出真实画面（spotlight 效果）。",
        "顶部新增『关于』按钮，可查看每次更新的功能。",
        "工具内实时显示命中状态（命中徽标 + 最近命中对象/时间/匹配度）。",
        "检测图案支持预览（列表图标 + 详情缩略图）。",
    ]),
    ("v6.1", "问题修复", [
        "修复 『+ 框选添加』失效：覆盖层误用 Qt.SubWindow 导致不显示，改为顶层工具窗 + paintEvent 绘制。",
        "修复 『抓取该应用窗口』失效：多进程应用窗口由 helper 进程拥有，改用 tier 回退（同包 pid 群 / 窗口名 / 兜底）。",
    ]),
    ("v6.0", "核心功能", [
        "响铃改为状态驱动：图案出现响铃，消失自动停铃，也可手动停止。",
        "支持多个检测图案，并可增/删/改名/启停/预览/重置学习（不再只有一个、不一键清空）。",
        "自学习：每次响铃可标记『命中/误报』，误报抬高阈值、命中精炼模板，越用越准。",
        "可选运行中的应用，自动抓取其窗口实时界面做检测。",
    ]),
    ("v1~5", "早期版本", [
        "基础框选监控、模板匹配、提示音报警、多图案管理雏形。",
    ]),
]

# ---------------------------------------------------------------------------
# 原生提示音（NSSound）
# ---------------------------------------------------------------------------

class SoundPlayer:
    def __init__(self):
        self._sounds = self._discover()
        self._current = None   # 当前正在播放的 NSSound 实例（用于停掉上一个、防止声音叠加）

    @staticmethod
    def _discover():
        names = set()
        for d in ["/System/Library/Sounds", os.path.expanduser("~/Library/Sounds")]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".aiff", ".wav", ".caf", ".mp3")):
                        names.add(os.path.splitext(f)[0])
        return sorted(names) or ["Ping"]

    @property
    def names(self):
        return self._sounds

    @staticmethod
    def _resolve_path(name: str):
        for d in ["/System/Library/Sounds", os.path.expanduser("~/Library/Sounds")]:
            if not os.path.isdir(d):
                continue
            for ext in (".aiff", ".wav", ".caf", ".mp3"):
                p = os.path.join(d, name + ext)
                if os.path.exists(p):
                    return p
        return None

    def play(self, name: str, custom_path: str = ""):
        try:
            from Cocoa import NSSound
            path = custom_path if custom_path and os.path.isfile(custom_path) else self._resolve_path(name)
            # 先停掉上一个仍在播放的声音，杜绝连续响铃时多个声音叠加（“提示音重叠”）。
            # 注意：这里每次仍新建实例再播放（v6.19 的修复——共享实例连续 play 偶发不发声），
            # 只是额外在播新的之前把“上一个”实例停掉，二者不冲突。
            if self._current is not None:
                try:
                    self._current.stop()
                except Exception:
                    pass
                self._current = None
            if path and os.path.isfile(path):
                # 每次播放都新建 NSSound 实例从文件载入，规避 NSSound 共享实例
                # 连续 play() 偶发不发声的问题（导致『只响第一下/不持续响铃』）。
                snd = NSSound.alloc().initWithContentsOfFile_byReference_(path, False)
                if snd is not None:
                    snd.play()
                    self._current = snd
                    return
            # 回退：命名共享音（部分环境无独立音频文件）
            snd = NSSound.soundNamed_(name)
            if snd is not None:
                snd.stop()
                snd.play()
                self._current = snd
        except Exception as e:
            print("播放提示音失败:", e)

    def stop(self):
        """立即停止当前正在播放的提示音（用于『停止响铃』）。"""
        try:
            if self._current is not None:
                self._current.stop()
                self._current = None
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 框选覆盖层（透明、多屏、拖拽选区，返回全局点坐标）
# ---------------------------------------------------------------------------

class _ScreenOverlay(QWidget):
    """每屏一个全屏遮罩层，接收鼠标拖拽，绘出半透明选区（spotlight）。"""

    def __init__(self, screen, manager, bg_pixmap=None):
        super().__init__()
        self.screen = screen
        self.manager = manager
        self.bg_pixmap = bg_pixmap
        # 顶层工具窗。不要 Qt.SubWindow（需 MDI 父容器），否则不显示。
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.Window
        )
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.CrossCursor)
        self._geom = screen.geometry()
        self.setGeometry(self._geom)
        self._start = None
        self._end = None
        self._active = False

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # 1. 整屏半透明遮罩（约 35% 黑），保证"外面被遮、里面是选区"
        p.fillRect(self.rect(), QColor(0, 0, 0, 90))
        # 2. 计算选区（覆盖层局部坐标）
        sel = None
        if self._start and self._end:
            gx1, gy1 = self._start.x(), self._start.y()
            gx2, gy2 = self._end.x(), self._end.y()
            x1 = min(gx1, gx2) - self._geom.x()
            y1 = min(gy1, gy2) - self._geom.y()
            x2 = max(gx1, gx2) - self._geom.x()
            y2 = max(gy1, gy2) - self._geom.y()
            sel = QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        if sel is not None and sel.width() > 0 and sel.height() > 0:
            # 选区内：用截取的屏幕背景"挖洞"，露出真实画面（spotlight）
            if self.bg_pixmap and not self.bg_pixmap.isNull():
                p.drawPixmap(sel, self.bg_pixmap, sel)
            # 选区内淡红高亮，确保选区位置一目了然
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 59, 48, 35))
            p.drawRect(sel)
            # 红色实线外框（粗，保证可见）
            p.setPen(QPen(QColor(255, 59, 48), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRect(sel.adjusted(-1, -1, 1, 1))
            # 红色虚线内框（白芯，确保任何背景下都醒目）
            p.setPen(QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.DashLine))
            p.drawRect(sel.adjusted(2, 2, -2, -2))
            # 选区尺寸标注（红色底白字）
            p.setPen(QColor(255, 255, 255))
            p.setBrush(QColor(255, 59, 48))
            p.setFont(QFont(self.font().family(), 11, QFont.Weight.Bold))
            p.drawText(sel.adjusted(2, -20, 0, 0), f" {int(x2 - x1)} × {int(y2 - y1)} ")
        # 3. 顶部提示文字（始终显示）
        p.setPen(QColor(235, 235, 235))
        p.setFont(QFont(self.font().family(), 11, QFont.Weight.Bold))
        p.drawText(self.rect(),
                   Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                   "拖拽框选区域　·　按 ESC 取消")
        p.end()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._active = False
            self.manager.finish(0, 0, 0, 0)
            return
        super().keyPressEvent(e)

    def mousePressEvent(self, e):
        self._active = True
        self._start = e.globalPosition().toPoint()
        self._end = self._start
        self.update()

    def mouseMoveEvent(self, e):
        if self._active:
            self._end = e.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if not self._active:
            return
        self._active = False
        p1, p2 = self._start, e.globalPosition().toPoint()
        x = min(p1.x(), p2.x()); y = min(p1.y(), p2.y())
        w = abs(p2.x() - p1.x()); h = abs(p2.y() - p1.y())
        self.manager.finish(x, y, w, h)


class SelectionOverlay:
    """管理多屏覆盖层，选完后通过 callback(points_rect) 返回。"""

    def __init__(self, parent=None, cap=None):
        self.parent = parent
        self.overlays = []
        self._cb = None
        self._activated = False
        self._cap = cap or get_manager()

    def _capture_bg(self, geom):
        """在显示遮罩前截取该屏画面作为背景，避免黑屏/白框问题。"""
        try:
            img = self._cap.capture_points(geom.x(), geom.y(), geom.width(), geom.height())
            if img is not None:
                pm = numpy_to_pixmap(img)
                if not pm.isNull():
                    return pm
        except Exception as e:
            print("截取框选背景失败:", e)
        return None

    def select(self, callback):
        self._cb = callback
        if self._activated:
            return  # 防双开
        self._activated = True
        screens = QApplication.screens()
        if not screens:
            QMessageBox.warning(self.parent, "提示", "未检测到屏幕，无法框选。")
            self._finish(None)
            return
        for sc in screens:
            bg = self._capture_bg(sc.geometry())
            ov = _ScreenOverlay(sc, self, bg)
            self.overlays.append(ov)
        # 显示全部并拉到最前、激活焦点
        for ov in self.overlays:
            ov.show()
            ov.raise_()
            ov.activateWindow()
            ov.setFocus()
        QApplication.processEvents()
        # 兜底：再触发一次 raise，防止动画未完成
        QTimer.singleShot(50, lambda: [ov.raise_() for ov in self.overlays])

    def finish(self, x, y, w, h):
        self._finish((x, y, w, h) if (w >= 5 and h >= 5) else None)

    def _finish(self, rect):
        for ov in self.overlays:
            try:
                ov.close()
            except Exception:
                pass
        self.overlays = []
        self._activated = False
        if self._cb:
            self._cb(rect)


# ---------------------------------------------------------------------------
# 检测节拍（主线程 QTimer 驱动）
# ---------------------------------------------------------------------------
# 说明：自 v6.19 起，检测不再使用后台工作线程（曾在部分 macOS 环境因 Vision OCR
# 静默停摆导致监控不稳 / 预览卡死）。改为两个独立 QTimer：
#   · detect_timer  —— 每『检测间隔』抓一帧，跑一次『测试文字检测』式 OCR + 图案匹配，命中即响铃（持续监控）。
#   · preview_timer —— 独立高频只抓帧刷新监控画面，保证实时流畅、不受检测节拍阻塞。



# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def numpy_to_pixmap(img: np.ndarray, maxw=None, maxh=None) -> QPixmap:
    if img is None:
        return QPixmap()
    # 兼容 float32 精炼模板 / 任意非 uint8 数组：先归一化到 0-255 再转 uint8
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = (img * 255.0)
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if maxw and maxh:
        scale = min(maxw / w, maxh / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 确保 rgb 是 C-连续，防止 QImage 因 strides 异常出现花屏/条纹
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

GLOBAL_QSS = """
QMainWindow { background: #f5f5f7; }
QLabel { color: #1d1d1f; }
QLabel#appTitle { font-size: 15px; font-weight: bold; color: #1d1d1f; }
QLabel#subText { color: #6e6e73; font-size: 11px; }
QLabel#hint { color: #86868b; font-size: 10px; }
QLabel#statusText { color: #1d1d1f; font-weight: bold; font-size: 12px; }
QLabel#fbTitle { font-weight: bold; color: #1d1d1f; font-size: 13px; }
QLabel#hitBadge { font-size: 16px; font-weight: bold; }
QLabel#fbThumb, QLabel#previewWell { background: #1c1c1e; color: #8e8e93; border-radius: 6px; }
QGroupBox {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 10px;
    margin-top: 20px;
    padding: 14px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px; top: 0px;
    padding: 3px 8px;
    background: #ffffff;
    color: #1d1d1f;
    font-weight: bold;
    font-size: 13px;
    font-size: 12px;
}
QGroupBox#feedbackCard {
    background: #ffffff;
    border: 1px solid #c7c7cc;
    border-radius: 10px;
    max-width: 300px;
}
QPushButton {
    background: #e8e8ed;
    color: #1d1d1f;
    border: none;
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 12px;
}
QPushButton:hover { background: #dcdce1; }
QPushButton:pressed { background: #cfcfd6; }
QPushButton:disabled { background: #f0f0f3; color: #b0b0b8; }
QPushButton[class="primary"] { background: #0071e3; color: #ffffff; font-weight: bold; }
QPushButton[class="primary"]:hover { background: #0077ed; }
QPushButton[class="success"] { background: #34c759; color: #ffffff; font-weight: bold; }
QPushButton[class="success"]:hover { background: #30d35f; }
QPushButton[class="danger"] { background: #ff3b30; color: #ffffff; font-weight: bold; }
QPushButton[class="danger"]:hover { background: #ff453a; }
QLineEdit, QComboBox, QSpinBox, QListWidget {
    background: #ffffff;
    border: 1px solid #c7c7cc;
    border-radius: 6px;
    padding: 4px 6px;
    color: #1d1d1f;
    selection-background-color: #0071e3;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #0071e3; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; border: none; }
QListWidget { padding: 3px; outline: none; }
QListWidget::item:selected { background: #e5f1ff; color: #1d1d1f; border-radius: 5px; }
QCheckBox { spacing: 5px; color: #1d1d1f; font-size: 12px; }
QSlider::groove:horizontal { background: #d2d2d7; height: 4px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #0071e3; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; border: 1px solid #b0b0b8; width: 12px; height: 12px; border-radius: 6px; margin-top: -4px; margin-bottom: -4px; }
/* 检测状态分组 387×140，单独收紧标题区与内边距，内部内容宽松不重叠 */
QGroupBox#statusGroup { margin-top: 12px; padding: 8px 6px 6px 6px; }
QGroupBox#statusGroup::title { top: 0px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = Detector()
        self.cap = get_manager()
        self.store = ConfigStore()
        self._ready = False   # 初始化守卫：init_ui 期间信号触发的保存会被跳过，避免用默认值冲掉用户存档
        self.sound = SoundPlayer()
        self.region: Optional[tuple] = None
        self.detect_timer = QTimer(self)   # 检测节拍：每『检测间隔』跑一次检测+响铃（持续监控）
        self.detect_timer.timeout.connect(self._detect_step)
        self.preview_timer = QTimer(self)  # 预览节拍：独立高频刷新监控画面（不影响检测）
        self.preview_timer.timeout.connect(self._preview_step)
        self._last_recognized = None
        self._last_matched_texts = set()
        self._last_results = []
        self.alerting = False
        self.beep_timer = None
        self.current_feedback = None   # (tid, score, crop)
        self._suppress_tpl_item_changed = False  # 重建列表时屏蔽 itemChanged
        self.overlay = None
        self._last_debug = {}
        self.paused = False          # 暂停响铃：手动暂停/反馈后静音
        # 响铃恢复方式（三态）：0=手动恢复（点『启动响铃』）；1=下次新命中自动恢复（默认）；2=暂停 N 秒后自动解除暂停
        self.ring_resume_mode = 1
        self.ring_resume_seconds = 10  # mode==2 时的自动解除秒数（用户可自定义 1-600）
        self._resume_timer = None    # N 秒自动解除暂停用的 QTimer（仅 mode==2 时启用）
        self._prev_matched = False   # 上一帧 matched 状态，用于识别「新一轮命中（上升沿）」以支撑自动恢复
        self._ring_sound = "Ping"     # 当前响铃使用的声音名（按命中来源确定）
        self._ring_custom = ""        # 当前响铃使用的自定义声音路径
        self._text_sounds = {}        # 各文字独立提示音映射：{文字: 声音名, ...}（空值=用最下方全局默认）
        self._detecting = False       # 是否处于检测中（用于丢弃停止后的迟到 tick）
        self._last_text_diag_log = 0  # 文字模式未命中诊断日志节流（秒时间戳）
        # 遮挡自恢复状态：物理遮挡（窗口盖住监控区域 / 全屏切换 / 抓取暂不可用）时，
        # 监控必须持续重试、绝不中断；记录遮挡起始时刻，用于状态栏显示"已重试 Xs"与起止日志。
        self._occlusion_since = None  # None 表示当前未遮挡
        # 文字检测去抖状态：OCR 不是每帧都稳定，连续命中若干帧才确认、连续缺失若干帧才清除
        self._text_hit_streak = 0
        self._text_miss_streak = 0
        self._text_detected = False
        # 监控预览视图状态：scale=1.0 表示完整适配 QLabel；offset 为像素平移
        self._preview_scale = 1.0
        # 框选完成后预览默认缩放倍数：等价于连点 3 下「－」缩小（0.8³ ≈ 0.512），留出整体视野
        self._preview_default_scale = 0.8 ** 3
        self._preview_offset = QPoint(0, 0)
        self._preview_base_pixmap: Optional[QPixmap] = None
        self._init_ui()
        self._load_config()
        # 主动申请『屏幕录制』权限（触发系统授权弹窗；已授权为 no-op）。
        # 这是修复『窗口遮挡后检测失效』的关键：未授权时 macOS 会把其它 App 窗口
        # 在截图里涂成纯色/空白，导致监控区域一旦被遮挡就检测不到。
        try:
            request_screen_capture_access()
        except Exception as e:  # pragma: no cover
            print("申请屏幕录制权限异常:", e)
        # 启动后稍延迟检查权限状态并提示（避免遮挡即失效却无提示）
        QTimer.singleShot(600, self._check_screen_permission)
        self._ready = True   # 初始化完成：此后用户操作/关闭时才真正落盘

    # ---------------- UI ----------------
    def _init_ui(self):
        self.setWindowTitle("框选屏幕检测工具 v6.48")
        # 用户授权：窗口 880（宽不变）×963（高），把增加的高度用于加高「检测状态」分组，保证显示正常
        self.setFixedSize(880, 963)

        cw = QWidget(); cw.setObjectName("centralWidget")
        self.setCentralWidget(cw)
        outer = QVBoxLayout(cw)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)

        # 顶部栏
        topbar = QHBoxLayout()
        title_lbl = QLabel("框选屏幕检测工具")
        title_lbl.setObjectName("appTitle")
        self.btn_about = QPushButton("关于")
        self.btn_about.setFixedWidth(60)
        topbar.addWidget(title_lbl)
        topbar.addStretch(1)
        topbar.addWidget(self.btn_about)
        outer.addLayout(topbar)

        # 主区域：左宽右窄
        root = QHBoxLayout()
        root.setSpacing(12)
        outer.addLayout(root, 1)

        # ---------- 左栏 ----------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        g_app = QGroupBox("监控区域")
        app_l = QVBoxLayout(g_app)
        row = QHBoxLayout()
        self.btn_region = QPushButton("① 框选监控区域")
        self.btn_region.setProperty("class", "primary")
        self.lbl_region = QLabel("未设置监控区域")
        self.lbl_region.setObjectName("subText")
        row.addWidget(self.btn_region)
        row.addWidget(self.lbl_region, 1)
        app_l.addLayout(row)
        lv.addWidget(g_app)

        g_mon = QGroupBox("监控预览")
        mon_l = QVBoxLayout(g_mon)
        well = QWidget()
        well_grid = QGridLayout(well)
        well_grid.setContentsMargins(0, 0, 0, 0)
        well_grid.setSpacing(0)
        self.preview = QLabel("开始检测后显示实时画面")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(432, 160)
        self.preview.setObjectName("previewWell")
        well_grid.addWidget(self.preview, 0, 0)
        g_feedback = QGroupBox("命中确认")
        g_feedback.setObjectName("feedbackCard")
        fb_l = QVBoxLayout(g_feedback)
        fb_l.setContentsMargins(10, 10, 10, 10)
        fb_l.setSpacing(6)
        self.fb_title = QLabel("检测到图案")
        self.fb_title.setObjectName("fbTitle")
        fb_l.addWidget(self.fb_title)
        fb_thumb_row = QHBoxLayout()
        self.fb_thumb = QLabel()
        self.fb_thumb.setFixedSize(80, 54)
        self.fb_thumb.setAlignment(Qt.AlignCenter)
        self.fb_thumb.setObjectName("fbThumb")
        fb_thumb_row.addWidget(self.fb_thumb)
        self.fb_info = QLabel("匹配度 - 请点击下方按钮确认")
        self.fb_info.setObjectName("subText")
        self.fb_info.setWordWrap(True)
        fb_thumb_row.addWidget(self.fb_info, 1)
        fb_l.addLayout(fb_thumb_row)
        fb_btns = QHBoxLayout()
        self.fb_hit = QPushButton("✓ 命中")
        self.fb_hit.setProperty("class", "success")
        self.fb_miss = QPushButton("✗ 误报")
        self.fb_miss.setProperty("class", "danger")
        self.fb_hit.clicked.connect(lambda: self._on_feedback(True))
        self.fb_miss.clicked.connect(lambda: self._on_feedback(False))
        fb_btns.addWidget(self.fb_hit)
        fb_btns.addWidget(self.fb_miss)
        fb_l.addLayout(fb_btns)
        g_feedback.hide()
        self.feedback_card = g_feedback
        well_grid.addWidget(g_feedback, 0, 0, Qt.AlignCenter)
        mon_l.addWidget(well, 0, Qt.AlignCenter)

        # 监控预览缩放/平移控制（紧凑一行，不影响整体布局）
        pv_ctrl = QHBoxLayout()
        pv_ctrl.setSpacing(4)
        pv_ctrl.setContentsMargins(0, 4, 0, 0)
        self.btn_pv_zoom_out = QPushButton("-")
        self.btn_pv_zoom_in = QPushButton("+")
        self.btn_pv_left = QPushButton("<")
        self.btn_pv_up = QPushButton("^")
        self.btn_pv_down = QPushButton("v")
        self.btn_pv_right = QPushButton(">")
        self.btn_pv_reset = QPushButton("重")
        for btn in (self.btn_pv_zoom_out, self.btn_pv_zoom_in,
                    self.btn_pv_left, self.btn_pv_up, self.btn_pv_down,
                    self.btn_pv_right, self.btn_pv_reset):
            btn.setFixedHeight(22)
            btn.setFixedWidth(26)
        self.btn_pv_reset.setFixedWidth(50)
        pv_ctrl.addStretch(1)
        pv_ctrl.addWidget(self.btn_pv_zoom_out)
        pv_ctrl.addWidget(self.btn_pv_zoom_in)
        pv_ctrl.addSpacing(6)
        pv_ctrl.addWidget(self.btn_pv_left)
        pv_ctrl.addWidget(self.btn_pv_up)
        pv_ctrl.addWidget(self.btn_pv_down)
        pv_ctrl.addWidget(self.btn_pv_right)
        pv_ctrl.addSpacing(6)
        pv_ctrl.addWidget(self.btn_pv_reset)
        pv_ctrl.addStretch(1)
        mon_l.addLayout(pv_ctrl)
        self.btn_pv_zoom_in.clicked.connect(lambda: self._zoom_preview(1.2))
        self.btn_pv_zoom_out.clicked.connect(lambda: self._zoom_preview(0.833))
        self.btn_pv_left.clicked.connect(lambda: self._pan_preview(-20, 0))
        self.btn_pv_up.clicked.connect(lambda: self._pan_preview(0, -20))
        self.btn_pv_down.clicked.connect(lambda: self._pan_preview(0, 20))
        self.btn_pv_right.clicked.connect(lambda: self._pan_preview(20, 0))
        self.btn_pv_reset.clicked.connect(self._reset_preview_view)
        lv.addWidget(g_mon)

        # 用户要求：把「运行日志」从右栏移到左栏（原检测图案位置）；宽度自适应撑满左栏
        g_log = QGroupBox("运行日志")
        logl = QVBoxLayout(g_log)
        # 用户授权：运行日志固定 215 高度；移到左栏后宽度自适应撑满，不再固定 387
        g_log.setFixedHeight(215)
        g_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.log = QListWidget()
        self.log.setMinimumHeight(120)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        logl.addWidget(self.log)
        lv.addWidget(g_log)

        # ---------- 右栏 ----------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        g_text = QGroupBox("检测模式 / 文字")
        tx_l = QVBoxLayout(g_text)
        mode_row = QHBoxLayout()
        self.combo_match_mode = QComboBox()
        self.combo_match_mode.addItems(["仅图案", "仅文字", "文字优先·图案兜底"])
        self.combo_match_mode.setCurrentIndex(1)
        self.combo_match_mode.setToolTip("默认『仅文字』：开始检测后按检测间隔持续做 OCR 文字检测。文字优先·图案兜底：优先文字检测，文字检测不到目标时再用图案命中兜底（文字或图案任一命中即响铃）。")
        mode_row.addWidget(QLabel("匹配:"))
        mode_row.addWidget(self.combo_match_mode, 1)
        tx_l.addLayout(mode_row)
        tx_row1 = QHBoxLayout()
        self.le_target_text = QLineEdit()
        self.le_target_text.setPlaceholderText("输入要检测的文字，多个用逗号/、隔开")
        tx_row1.addWidget(QLabel("目标:"))
        tx_row1.addWidget(self.le_target_text, 1)
        tx_l.addLayout(tx_row1)
        tx_row2 = QHBoxLayout()
        self.combo_text_mode = QComboBox()
        self.combo_text_mode.addItems(["包含任一", "全部包含", "完全相等"])
        self.btn_test_text = QPushButton("测试文字检测")
        tx_row2.addWidget(QLabel("文字匹配:"))
        tx_row2.addWidget(self.combo_text_mode)
        tx_row2.addWidget(self.btn_test_text)
        tx_l.addLayout(tx_row2)
        tx_l.addWidget(QLabel("各文字提示音（每行一个文字）:"))
        self.text_snd_list = QListWidget()
        # 每次只完整显示一行文字提示音，其余项通过垂直滚动条查看
        # 窗口加高后，文字提示音列表多留一点可视行
        self.text_snd_list.setFixedHeight(70)
        self.text_snd_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_snd_list.setToolTip("为每个文字单独指定提示音；未指定则用最下方全局默认提示音")
        tx_l.addWidget(self.text_snd_list)
        self._text_snd_combos = {}
        # 给「匹配/目标/文字匹配」三行足够高度，避免标签/输入框被截断；
        # 为检测图案详情区（改为垂直布局）腾出纵向空间，适度压缩本区。
        g_text.setMinimumHeight(170)
        rv.addWidget(g_text)

        g_param = QGroupBox("检测节奏 / 参数")
        p_l = QVBoxLayout(g_param)
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel("间隔(s):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 300)
        self.spin_interval.setValue(1)
        self.spin_interval.setToolTip("每多少秒执行一次检测（OCR 文字检测 + 图案匹配）")
        p_row.addWidget(self.spin_interval)
        p_row.addWidget(QLabel("冷却(s):"))
        self.spin_cooldown = QSpinBox()
        self.spin_cooldown.setRange(1, 60)
        self.spin_cooldown.setValue(3)
        p_row.addWidget(self.spin_cooldown)
        p_row.addStretch(1)
        self.chk_ms = QCheckBox("多尺度匹配")
        self.chk_ms.setChecked(True)
        p_row.addWidget(self.chk_ms)
        p_l.addLayout(p_row)
        rv.addWidget(g_param)

        g_snd = QGroupBox("默认提示音")
        # 压缩默认提示音分组高度，把空间让给改为垂直布局的检测图案详情区。
        g_snd.setFixedSize(387, 70)
        s_l = QHBoxLayout(g_snd)
        s_l.setContentsMargins(8, 10, 8, 8)
        s_l.setSpacing(6)
        self.snd_combo = QComboBox()
        self.snd_combo.addItems(self.sound.names + ["自定义…"])
        self.snd_combo.setToolTip("全局默认提示音。当图案/文字均未单独指定提示音时，一律以此声音响铃。")
        # 固定下拉高度：避免 85px 总高内被布局压缩导致文字截断
        self.snd_combo.setFixedHeight(22)
        self.btn_browse_sound = QPushButton("浏览…")
        self.btn_test = QPushButton("测试")
        self.btn_browse_sound.setFixedHeight(22)
        self.btn_test.setFixedHeight(22)
        s_l.addWidget(self.snd_combo, 1)
        s_l.addWidget(self.btn_browse_sound)
        s_l.addWidget(self.btn_test)
        self._custom_sound_path = ""
        rv.addWidget(g_snd)

        g_status = QGroupBox("检测状态")
        self.g_status = g_status
        g_status.setObjectName("statusGroup")
        # 检测状态保持足够高度；为检测图案详情区（垂直布局）适度让出空间。
        g_status.setFixedSize(387, 155)
        st_l = QVBoxLayout(g_status)
        # 387×165 包含分组标题与边框；样式表已单独为 #statusGroup 收紧标题/边距
        st_l.setContentsMargins(6, 8, 6, 8)
        st_l.setSpacing(5)
        self.hit_badge = QLabel("待机")
        self.hit_badge.setAlignment(Qt.AlignCenter)
        self.hit_badge.setObjectName("hitBadge")
        self.hit_badge.setWordWrap(False)
        # 命中/监控中状态：16px 粗体 + 4px*2 padding = 24px，min 30px 给文字上下各留 3px
        self.hit_badge.setMinimumHeight(30)
        self.hit_badge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        st_l.addWidget(self.hit_badge)
        self.hit_detail = QLabel("开始检测后，这里实时显示是否命中及命中对象")
        self.hit_detail.setObjectName("subText")
        self.hit_detail.setWordWrap(True)
        # 用 margin-top/bottom 替代 addSpacing；#subText 字号 11px，
        # 预留 36px 可完整显示两行说明文字，避免「未命中」多行文本被压缩
        self.hit_detail.setStyleSheet("margin-top:5px;margin-bottom:4px;")
        self.hit_detail.setFixedHeight(36)
        st_l.addWidget(self.hit_detail)
        ring_h = QHBoxLayout()
        ring_h.setSpacing(6)
        ring_h.addWidget(QLabel("恢复方式："))
        self.combo_ring_resume = QComboBox()
        self.combo_ring_resume.addItems([
            "手动恢复",
            "新命中自动恢复",
            "自定义秒数后恢复",
            "命中跟随（命中响/消失静音）",
        ])
        self.combo_ring_resume.setCurrentIndex(1)
        self.combo_ring_resume.setToolTip(
            "暂停响铃后的恢复方式：\n"
            "• 手动恢复：暂停后必须手动点『启动响铃』才恢复。\n"
            "• 新命中自动恢复（默认）：目标消失再出现（新一轮命中）会自动解除暂停并响铃，不漏新事件。\n"
            "• 自定义秒数后恢复：暂停起 N 秒倒计时，到点自动解除暂停；右侧数字可自定义 1-600 秒。\n"
            "• 命中跟随：响铃严格跟随命中状态——命中时自动播放，命中消失时自动静音；"
            "手动暂停后，下一次命中会自动恢复响铃。")
        # 加宽下拉，避免「自定义秒数后恢复」等长选项在弹出列表中被截断
        self.combo_ring_resume.setMinimumWidth(180)
        # 140px 总高已有余量，下拉框 20px 显示更舒展
        self.combo_ring_resume.setMinimumHeight(20)
        self.combo_ring_resume.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        self.combo_ring_resume.currentIndexChanged.connect(self._on_ring_mode_changed)
        self.spin_ring_seconds = QSpinBox()
        self.spin_ring_seconds.setRange(1, 600)
        self.spin_ring_seconds.setValue(self.ring_resume_seconds)
        self.spin_ring_seconds.setFixedWidth(70)
        self.spin_ring_seconds.setMinimumHeight(20)
        self.spin_ring_seconds.setSuffix(" 秒")
        self.spin_ring_seconds.setToolTip("自定义自动解除暂停的秒数（仅对『自定义秒数后恢复』生效）")
        self.spin_ring_seconds.setEnabled(self.combo_ring_resume.currentIndex() == 2)
        self.spin_ring_seconds.valueChanged.connect(self._on_ring_seconds_changed)
        ring_h.addWidget(self.combo_ring_resume, 1)
        ring_h.addWidget(self.spin_ring_seconds)
        st_l.addLayout(ring_h)
        rv.addWidget(g_status)

        # 用户要求：把「检测图案」从左栏移到右栏（原运行日志位置）
        g_tpl = QGroupBox("检测图案")
        tpl_l = QVBoxLayout(g_tpl)
        self.tpl_list = QListWidget()
        # 列表适度压缩，为垂直布局的详情区腾出空间
        self.tpl_list.setFixedHeight(50)
        self.tpl_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.tpl_list.itemChanged.connect(self._on_tpl_item_changed)
        tpl_l.addWidget(self.tpl_list)
        self.tpl_hint = QLabel("勾选=参与检测；Cmd/Ctrl/Shift 多选批量删除/重置")
        self.tpl_hint.setObjectName("hint")
        self.tpl_hint.setStyleSheet("margin-top:0px;margin-bottom:0px;padding:0px;")
        self.tpl_hint.setFixedHeight(18)
        tpl_l.addWidget(self.tpl_hint)
        tpl_btns = QHBoxLayout()
        self.btn_add = QPushButton("＋ 框选添加")
        self.btn_load = QPushButton("从图片载入")
        self.btn_del = QPushButton("删除选中")
        self.btn_reset = QPushButton("重置学习")
        tpl_btns.addWidget(self.btn_add)
        tpl_btns.addWidget(self.btn_load)
        tpl_btns.addWidget(self.btn_del)
        tpl_btns.addWidget(self.btn_reset)
        tpl_l.addLayout(tpl_btns)
        self.detail = QWidget()
        # 详情区改为「预览在上、表单在下」的垂直布局，需要更多高度；
        # 宽度不再受挤压，文字/按钮不再截断重叠。
        self.detail.setMinimumHeight(170)
        tpl_l.addWidget(self.detail)
        self._build_detail()
        rv.addWidget(g_tpl, 1)

        root.addWidget(left, 5)
        root.addWidget(right, 4)

        # ---------- 底栏 ----------
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.btn_start = QPushButton("▶ 开始检测")
        self.btn_start.setProperty("class", "primary")
        self.btn_ring = QPushButton("⏸ 暂停响铃")
        self.btn_ring.setEnabled(False)
        bottom.addWidget(self.btn_start, 2)
        bottom.addWidget(self.btn_ring, 2)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("statusText")
        bottom.addWidget(self.lbl_status, 3)
        outer.addLayout(bottom)

        # 信号
        self.btn_region.clicked.connect(lambda: self._begin_select(self._set_region))
        self.btn_add.clicked.connect(lambda: self._begin_select(self._add_template))
        self.btn_load.clicked.connect(self._load_image)
        self.btn_del.clicked.connect(self._del_template)
        self.btn_reset.clicked.connect(self._reset_learning)
        self.tpl_list.currentItemChanged.connect(self._on_select_template)
        self.btn_start.clicked.connect(self._toggle_detect)
        self.btn_ring.clicked.connect(self._on_ring_button_clicked)
        self.btn_test.clicked.connect(self._test_sound)
        self.btn_browse_sound.clicked.connect(self._browse_custom_sound)
        self.snd_combo.currentIndexChanged.connect(self._on_sound_selection_changed)
        self.btn_test_text.clicked.connect(self._test_text_detect)
        self.btn_about.clicked.connect(self._show_about)
        self.le_target_text.textChanged.connect(self._on_target_text_changed)
        self.combo_text_mode.currentIndexChanged.connect(lambda: self._save_config())
        if not self.cap.ready:
            self.lbl_status.setText("⚠ 无法初始化屏幕采集，请检查环境")
        elif not self.cap.quartz_available:
            self.lbl_status.setText("提示：未加载 pyobjc，窗口辅助抓取不可用；可正常使用『① 框选监控区域』。")
        self.setStyleSheet(GLOBAL_QSS)

    def _build_detail(self):
        # 检测图案详情区改为垂直布局：预览图在上、表单在下，
        # 解决右栏窄宽度下水平布局导致的文字/按钮截断重叠问题。
        l = QVBoxLayout(self.detail)
        l.setContentsMargins(0, 2, 0, 0)
        l.setSpacing(4)

        # 上方：图案预览（占满右栏宽度，高度固定，避免与表单抢水平空间）
        self.det_preview = QLabel("（选中后预览）")
        self.det_preview.setFixedHeight(74)
        self.det_preview.setMinimumWidth(200)
        self.det_preview.setAlignment(Qt.AlignCenter)
        self.det_preview.setStyleSheet("background:#1b1b1b;border:1px solid #444;color:#888;")
        l.addWidget(self.det_preview)

        # 下方：表单（独占整栏宽度，不再有水平挤压）
        form = QVBoxLayout()
        form.setSpacing(3)
        form.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QLabel("名称:"))
        self.det_name = QLineEdit()
        self.det_name.setFixedHeight(20)
        row1.addWidget(self.det_name, 1)
        self.det_enabled = QCheckBox("参与检测")
        self.det_enabled.setChecked(True)
        row1.addWidget(self.det_enabled)
        form.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(QLabel("基准阈值:"))
        self.det_thr = QSlider(Qt.Horizontal)
        self.det_thr.setRange(20, 99)
        self.det_thr.setValue(75)
        self.det_thr_val = QLabel("0.75")
        row2.addWidget(self.det_thr, 1)
        row2.addWidget(self.det_thr_val)
        form.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        self.det_eff = QLabel("有效阈值: -")
        self.det_stats = QLabel("学习: 命中0 误报0")
        row3.addWidget(self.det_eff)
        row3.addWidget(self.det_stats, 1)
        form.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(4)
        self.det_live = QLabel("当前最佳匹配度: -")
        self.det_live.setStyleSheet("color:#1565c0;")
        self.btn_test_match = QPushButton("测试匹配")
        self.btn_test_match.setToolTip("立即用当前模板匹配一次监控区域，并把结果写入日志")
        self.btn_save_img = QPushButton("保存图片")
        self.btn_save_img.setToolTip("把当前选中模板导出为 PNG 保存到任意位置")
        self.btn_test_match.setFixedHeight(20)
        self.btn_save_img.setFixedHeight(20)
        row4.addWidget(self.det_live, 1)
        row4.addWidget(self.btn_test_match)
        row4.addWidget(self.btn_save_img)
        form.addLayout(row4)

        row5 = QHBoxLayout()
        row5.setSpacing(4)
        self.det_sound = QComboBox()
        self.det_sound.addItems(["默认(全局)"] + self.sound.names)
        self.det_sound.setToolTip("为该图案单独指定提示音；选『默认(全局)』则使用全局声音")
        self.det_sound.setFixedHeight(20)
        row5.addWidget(QLabel("图案提示音:"))
        row5.addWidget(self.det_sound, 1)
        form.addLayout(row5)

        l.addLayout(form, 1)

        self.det_thr.valueChanged.connect(lambda v: self.det_thr_val.setText(f"{v/100:.2f}"))
        self.det_name.editingFinished.connect(self._apply_detail)
        self.det_enabled.toggled.connect(self._apply_detail)
        self.det_thr.valueChanged.connect(self._apply_detail)
        self.btn_test_match.clicked.connect(self._test_match_template)
        self.btn_save_img.clicked.connect(self._save_template_image)
        self.det_sound.currentIndexChanged.connect(self._on_detail_sound_changed)

    def _show_about(self):
        """关于弹窗：展示各版本更新功能。"""
        from PySide6.QtWidgets import QDialog, QTextEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("关于 · 框选屏幕检测工具")
        dlg.setMinimumSize(520, 420)
        v = QVBoxLayout(dlg)
        intro = QLabel(
            "<b>框选屏幕检测工具 v6.48</b><br>"
            "框选屏幕/窗口区域，截取图案作为模板，持续监控；"
            "图案出现即播放提示音，并可对每个响铃标记『命中/误报』以自学习降误判。")
        intro.setWordWrap(True)
        v.addWidget(intro)
        te = QTextEdit()
        te.setReadOnly(True)
        html = ""
        for ver, title, items in CHANGELOG:
            html += f"<h3 style='margin:8px 0 4px;color:#1565c0;'>{ver} · {title}</h3><ul style='margin:0 0 6px;'>"
            for it in items:
                html += f"<li>{it}</li>"
            html += "</ul>"
        te.setHtml(html)
        v.addWidget(te, 1)
        close = QPushButton("关闭")
        close.clicked.connect(dlg.accept)
        v.addWidget(close)
        dlg.exec()

    def _set_hit_status(self, badge_text: str, badge_color: str, detail: str):
        """工具内实时显示命中状态（徽标 + 说明）。"""
        self.hit_badge.setText(badge_text)
        self.hit_badge.setStyleSheet(
            f"font-size:16px;font-weight:bold;color:#fff;background:{badge_color};"
            f"padding:4px 8px;border-radius:8px;")
        self.hit_detail.setText(detail)
        # 样式会改变 badge 的 sizeHint，必须刷新布局，否则说明文字可能贴到徽章上
        self.hit_badge.adjustSize()
        self.hit_badge.updateGeometry()
        self.g_status.layout().activate()

    def _set_ring_button(self, state: str):
        """统一管理『暂停/启动响铃』按钮的视觉与可用状态。
        state: 'idle' 监控中未响铃 | 'ringing' 正在响铃 | 'paused' 已暂停（可启动） | 'stopped' 未检测
        按钮标签随 self.paused 在『⏸ 暂停响铃』与『▶ 启动响铃』之间切换；
        响铃中高亮橙色，暂停态高亮蓝色，待命/禁用态灰色。"""
        btn = self.btn_ring
        if state == "stopped":
            btn.setText("⏸ 暂停响铃")
            btn.setStyleSheet("background:#9e9e9e;color:#fff;padding:8px;")
            btn.setEnabled(False)
            return
        if self.paused:
            btn.setText("▶ 启动响铃")
            btn.setStyleSheet("background:#0071e3;color:#fff;padding:8px;font-weight:bold;")
        else:
            btn.setText("⏸ 暂停响铃")
            if state == "ringing":
                btn.setStyleSheet("background:#ff9500;color:#fff;padding:8px;font-weight:bold;")
            else:  # idle：检测进行中但未响铃
                btn.setStyleSheet("background:#aeaeb2;color:#fff;padding:8px;")
        btn.setEnabled(True)

    def _load_template_preview(self, t):
        """在详情区显示模板缩略图（精炼优先）。"""
        img = None
        if t is not None:
            try:
                img = t.active_image
            except Exception:
                img = None
        if img is None:
            self.det_preview.setText("（无预览）")
            self.det_preview.setPixmap(QPixmap())
            return
        # 详情区预览图现在占满右栏宽度（约 340px），按预览区尺寸缩放。
        pm = numpy_to_pixmap(img, 340, 74)
        self.det_preview.setPixmap(pm)

    # ---------------- 配置 ----------------
    def _load_config(self):
        data = self.store.load()
        # 启动即清理『模板已删除但仍残留的孤立图片 / 上次异常遗留的 *.tmp』，
        # 防止长期运行后配置目录堆积无用文件占用存储（v6.24 临时文件自动清理）。
        # 注意：当存档模板数组为空（配置损坏/初次使用）时，绝不删除任何模板图片，
        # 否则会把用户真实模板当成孤儿一次性清空（见 config_store.gc_orphans 的保护）。
        try:
            n = self.store.gc_orphans()
            if n:
                self._log(f"已清理 {n} 个孤立/临时文件（配置目录）")
        except Exception:
            pass
        self.region = data.get("region")
        if self.region:
            self._show_region()
        # 兼容旧版毫秒配置：新版优先读 interval_sec；旧版 interval_ms 折算为秒（至少 1s）
        interval_sec = data.get("interval_sec")
        if interval_sec is None:
            interval_sec = max(1, data.get("interval_ms", 1000) // 1000)
        self.spin_interval.setValue(interval_sec)
        self.spin_cooldown.setValue(data.get("cooldown_s", 3))
        self.chk_ms.setChecked(data.get("multiscale", True))
        snd = data.get("sound", "Ping")
        self._custom_sound_path = data.get("custom_sound_path", "") or ""
        if self._custom_sound_path and os.path.isfile(self._custom_sound_path):
            self.snd_combo.setCurrentText("自定义…")
        elif snd in self.sound.names:
            self.snd_combo.setCurrentText(snd)
        else:
            self.snd_combo.setCurrentIndex(0)
        self.store.restore_templates(self.detector)
        self._refresh_tpl_list()
        # 自动保存（持久化）目标文字与文字匹配方式——之前仅模板被保存，
        # 文字目标重启即丢失；现在一并落盘，并在输入框/下拉变化时即时写入（输入即存）。
        self.le_target_text.setText(data.get("target_text", "") or "")
        self.combo_text_mode.setCurrentIndex(int(data.get("text_mode", 0) or 0))
        # 匹配模式（仅图案/仅文字/文字优先·图案兜底）一并落盘，重启后恢复上次选择
        _mm = data.get("match_mode", 1)
        self.combo_match_mode.setCurrentIndex(int(_mm) if isinstance(_mm, int) else 1)
        # 响铃恢复方式（三态）：默认 1=下次新命中自动恢复；兼容旧配置里的 ring_resume_auto 布尔
        _mode = data.get("ring_resume_mode", None)
        if _mode is None:
            _mode = 1 if bool(data.get("ring_resume_auto", True)) else 0
        self.ring_resume_mode = int(_mode) if _mode in (0, 1, 2, 3) else 1
        self.combo_ring_resume.setCurrentIndex(self.ring_resume_mode)
        # 自定义秒数：默认 10，范围 1-600
        _secs = data.get("ring_resume_seconds", 10)
        try:
            _secs = int(_secs)
        except Exception:
            _secs = 10
        self.ring_resume_seconds = max(1, min(600, _secs))
        self.spin_ring_seconds.setValue(self.ring_resume_seconds)
        self.spin_ring_seconds.setEnabled(self.ring_resume_mode == 2)
        # 逐文字提示音映射（{文字: 声音名}），未指定的文字回退最下方全局默认
        self._text_sounds = data.get("text_sounds", {}) or {}
        self._rebuild_text_sound_list()

    def _save_config(self):
        if not getattr(self, "_ready", True):
            return   # 初始化期间（init_ui 信号触发）不落盘，避免用默认值覆盖用户存档
        params = {
            "interval_sec": self.spin_interval.value(),
            "cooldown_s": self.spin_cooldown.value(),
            "multiscale": self.chk_ms.isChecked(),
            "sound": self.snd_combo.currentText(),
            "custom_sound_path": getattr(self, "_custom_sound_path", "") or "",
            "target_text": self.le_target_text.text().strip(),
            "text_mode": self.combo_text_mode.currentIndex(),
            "match_mode": self.combo_match_mode.currentIndex(),
            "text_sounds": self._text_sounds,
            "ring_resume_mode": self.ring_resume_mode,
            "ring_resume_seconds": self.ring_resume_seconds,
        }
        self.store.save(self.detector, self.region, params)


    def _show_msg(self, title, text, icon=QMessageBox.Information):
        """稳健弹窗：避免 macOS 上 QMessageBox.information/warning 偶发不前置/不显示。

        关键处理：
          1. 不设置父窗口，使用独立顶层窗口，避免 macOS sheet 依附失败导致不显示。
          2. 强制 WindowStaysOnTopHint，确保弹窗一定在其它窗口之上。
          3. 主窗口 raise/activate + 弹窗 show/raise/activate 之间多次 processEvents，
             给窗口服务器足够时间创建并前置。
          4. 使用应用级模态 exec()，阻塞等待用户点击。
        """
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        box = QMessageBox()
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.setWindowFlags(
            Qt.Dialog
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTitleHint
            | Qt.WindowCloseButtonHint
        )
        # 注意：只通过 box.exec() 显示一次。之前这里额外调用了 box.show()，
        # 在 macOS 上 show() 已创建并显示原生窗口，随后 exec() 又显示一次，
        # 导致同一个提示框出现「两个窗口」。去掉 show() 后由 exec() 单次模态显示即可，
        # 既修复双窗口，也保持无父窗口的顶层模态行为（避免 macOS sheet 依附失败）。
        box.raise_()
        box.activateWindow()
        QApplication.processEvents()
        return box.exec()

    def _set_region(self, rect):
        if rect is None:
            return
        self.region = rect
        self._show_region()
        self._reset_preview_view()
        # 框选后默认把预览缩小到约 3 下「－」缩放，留出整体视野
        self._preview_scale = self._preview_default_scale
        # 框选后立即启动预览节拍（独立于检测），无需点『开始检测』即可看到实时画面
        if not self.preview_timer.isActive():
            self.preview_timer.setInterval(200)
            self.preview_timer.start()
        self._log(f"已框选监控区域: {rect}")

    def _show_region(self):
        if not self.region:
            return
        x, y, w, h = [int(v) for v in self.region]
        self.lbl_region.setText(f"区域: 起点({x},{y}) 尺寸 {w}×{h}")
        self.lbl_region.setStyleSheet("color:#2e7d32;")

    # ---------------- 框选 ----------------
    def _begin_select(self, callback):
        self.overlay = SelectionOverlay(self, self.cap)
        self.overlay.select(callback)

    # ---------------- 模板管理 ----------------
    def _refresh_tpl_list(self):
        self._suppress_tpl_item_changed = True
        self.tpl_list.blockSignals(True)
        self.tpl_list.clear()
        for t in self.detector.templates:
            item = QListWidgetItem(t.name)
            item.setData(Qt.UserRole, t.tid)
            # 勾选框：勾选=参与检测，取消勾选=本轮不参与（仍保留模板）
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if t.enabled else Qt.Unchecked)
            # 列表项预览图标
            try:
                pm = numpy_to_pixmap(t.active_image, 64, 64)
                if not pm.isNull():
                    item.setIcon(QIcon(pm))
            except Exception:
                pass
            self.tpl_list.addItem(item)
        self.tpl_list.blockSignals(False)
        self._suppress_tpl_item_changed = False
        if self.tpl_list.count():
            self.tpl_list.setCurrentRow(0)
        else:
            self._clear_detail()

    def _on_tpl_item_changed(self, item):
        """勾选框变化：开关该图案是否参与检测。默认全勾选=全部参与（等同旧行为）。"""
        if self._suppress_tpl_item_changed or item is None:
            return
        tid = item.data(Qt.UserRole)
        if tid is None:
            return
        self.detector.set_enabled(tid, item.checkState() == Qt.Checked)
        self._save_config()  # 持久化勾选状态

    def _add_template(self, rect):
        if rect is None:
            return
        x, y, w, h = rect
        img = self.cap.capture_points(x, y, w, h)
        if img is None:
            QMessageBox.warning(self, "失败", "截图失败，请确认已授予『屏幕录制』权限。")
            return
        name, ok = self._ask_name("新图案", f"图案{len(self.detector.templates)+1}")
        if not ok or not name:
            return
        t = self.detector.add_template(name, img)
        self.store.save_image(t.tid, img)
        self._refresh_tpl_list()
        self._save_config()
        self._log(f"已添加模板: {name} ({w}×{h})")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图案图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "失败", "无法读取图片。")
            return
        # 大图预警：整窗截图类模板远大于监控区域，几乎无法命中
        ih, iw = img.shape[:2]
        if max(iw, ih) > 1200:
            QMessageBox.information(self, "提示",
                f"载入图片较大（{iw}×{ih}）。若为整窗截图，模板会远大于监控区域而难以命中。\n"
                "建议：用『＋ 框选添加』直接从屏幕截取目标，或载入仅含目标的小图。")
        name, ok = self._ask_name("新图案", os.path.splitext(os.path.basename(path))[0])
        if not ok or not name:
            return
        t = self.detector.add_template(name, img)
        self.store.save_image(t.tid, img)
        # —— 自动校准：若已设监控区域，按当前画面把阈值调到"刚好能命中" ——
        # 载入图片常与屏上目标分辨率/渲染略有差异，固定阈值易漏检；
        # 用当前监控区域实测最佳得分，把基准阈值设为 得分-余量，保证能命中。
        if self.region:
            x, y, w, h = self.region
            frame = self.cap.capture_points(x, y, w, h)
            if frame is not None:
                dbg = self.detector.debug_match_scores(frame, only_tids=[t.tid])
                info = dbg.get(t.tid, {})
                sc = info.get("best_score", -1.0)
                if sc >= 0.55:
                    new_thr = max(0.50, min(0.95, sc - 0.04))
                    t.base_threshold = new_thr
                    self._log(f"已按当前监控区域自动校准『{name}』阈值: {new_thr:.2f}（最佳得分 {sc:.2f}）")
                else:
                    self._log(f"『{name}』与当前监控区域匹配度偏低({sc:.2f})，可能无法命中；"
                              f"可尝试『＋ 框选添加』或调低其基准阈值")
        else:
            self._log(f"未设置监控区域，『{name}』暂用默认阈值；设置区域后可点『测试匹配』校准")
        self._refresh_tpl_list()
        self._save_config()
        self._log(f"已从图片载入模板: {name}")

    def _del_template(self):
        items = self.tpl_list.selectedItems()
        if not items:
            items = [self.tpl_list.currentItem()]  # 兜底：单选时 currentItem
        if not items or items[0] is None:
            return
        tids = [it.data(Qt.UserRole) for it in items if it is not None]
        if not tids:
            return
        for tid in tids:
            self.detector.remove_template(tid)
            self.store.delete_image(tid)
            self.store.delete_image(tid + "_refined")
        self._refresh_tpl_list()
        self._save_config()
        self._log(f"已删除选中模板（共 {len(tids)} 个）")

    def _reset_learning(self):
        items = self.tpl_list.selectedItems()
        if not items:
            items = [self.tpl_list.currentItem()]
        if not items or items[0] is None:
            return
        tids = [it.data(Qt.UserRole) for it in items if it is not None]
        if not tids:
            return
        for tid in tids:
            self.detector.reset_learning(tid)
            self.store.delete_image(tid + "_refined")
        self._refresh_tpl_list()
        self._save_config()
        self._log(f"已重置所选模板的学习数据（共 {len(tids)} 个）")

    def _on_select_template(self, cur, prev):
        if not cur:
            return
        tid = cur.data(Qt.UserRole)
        t = self.detector.get(tid)
        if not t:
            return
        self.det_name.setText(t.name)
        self.det_enabled.setChecked(t.enabled)
        self.det_thr.setValue(int(round(t.base_threshold * 100)))
        self.det_thr_val.setText(f"{t.base_threshold:.2f}")
        self.det_eff.setText(f"有效阈值: {t.effective_threshold:.2f}")
        self.det_stats.setText(f"学习: 命中{len(t.pos_scores)} 误报{len(t.neg_scores)}"
                               + (" 已精炼" if t.refined_image is not None else ""))
        # 显示该模板在最新帧中的最佳得分
        info = self._last_debug.get(t.tid, {})
        if info:
            score = info.get('best_score', -1.0)
            thr = info.get('threshold', t.effective_threshold)
            self.det_live.setText(f"当前最佳匹配度: {score:.2f} {'✓' if score >= thr else '✗'}")
        else:
            self.det_live.setText("当前最佳匹配度: -")
        # 载入该图案的独立提示音（空=默认全局）
        if t.sound and t.sound in self.sound.names:
            self.det_sound.setCurrentText(t.sound)
        else:
            self.det_sound.setCurrentText("默认(全局)")
        self._load_template_preview(t)

    def _on_detail_sound_changed(self):
        """模板详情区的『图案提示音』变化：写入当前选中图案并即时持久化。"""
        item = self.tpl_list.currentItem()
        if not item:
            return
        tid = item.data(Qt.UserRole)
        t = self.detector.get(tid)
        if not t:
            return
        val = self.det_sound.currentText()
        t.sound = "" if val == "默认(全局)" else val
        self._save_config()

    def _on_target_text_changed(self):
        """目标文字输入框变化：输入即存，并重建逐文字提示音列表（保留已选声音）。"""
        self._save_config()
        self._rebuild_text_sound_list()

    def _rebuild_text_sound_list(self):
        """根据『目标文字』解析出的每个文字，重建逐文字提示音列表（每行一个文字 + 独立下拉）。"""
        self.text_snd_list.clear()
        self._text_snd_combos = {}
        texts = self._text_targets()
        if not texts:
            hint = QListWidgetItem("（先填写上方目标文字，即可为每个文字单独选声）")
            hint.setFlags(hint.flags() & ~Qt.ItemIsEnabled)
            hint.setForeground(QColor("#9aa3b2"))
            self.text_snd_list.addItem(hint)
            return
        vp_w = self.text_snd_list.viewport().width()
        avail = max(120, vp_w - 12) if vp_w > 0 else 180
        for t in texts:
            item = QListWidgetItem()
            widget = QWidget()
            widget.setFixedSize(avail, 34)
            row = QHBoxLayout(widget)
            row.setContentsMargins(6, 4, 6, 4)
            lbl = QLabel(t)
            lbl.setFixedHeight(22)
            lbl.setAlignment(Qt.AlignVCenter)
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            row.addWidget(lbl)
            combo = QComboBox()
            combo.setFixedHeight(22)
            combo.setMinimumWidth(90)
            items = ["默认(全局)"] + self.sound.names
            combo.addItems(items)
            cur = self._text_sounds.get(t, "")
            combo.setCurrentText(cur if cur in items else "默认(全局)")
            combo.currentTextChanged.connect(lambda v, _t=t: self._on_text_sound_changed(_t, v))
            row.addWidget(combo, 1)
            widget.setLayout(row)
            item.setSizeHint(QSize(avail, 34))
            self.text_snd_list.addItem(item)
            self.text_snd_list.setItemWidget(item, widget)
            self._text_snd_combos[t] = combo

    def _on_text_sound_changed(self, text, value):
        """逐文字提示音下拉变化：写入该文字的独立提示音并即时持久化。"""
        self._text_sounds[text] = "" if value == "默认(全局)" else value
        self._save_config()

    def _clear_detail(self):
        self.det_name.clear()
        self.det_stats.setText("学习: -")
        self.det_eff.setText("有效阈值: -")
        self.det_preview.setText("（选中后预览）")
        self.det_preview.setPixmap(QPixmap())

    def _apply_detail(self):
        item = self.tpl_list.currentItem()
        if not item:
            return
        tid = item.data(Qt.UserRole)
        t = self.detector.get(tid)
        if not t:
            return
        t.name = self.det_name.text() or t.name
        t.enabled = self.det_enabled.isChecked()
        t.base_threshold = self.det_thr.value() / 100.0
        item.setText(f"{'✓' if t.enabled else '✗'} {t.name}")
        self.det_thr_val.setText(f"{t.base_threshold:.2f}")
        self.det_eff.setText(f"有效阈值: {t.effective_threshold:.2f}")
        self._save_config()

    def _ask_name(self, title, default):
        from PySide6.QtWidgets import QInputDialog
        return QInputDialog.getText(self, title, "请输入图案名称:", text=default)

    # ---------------- 检测控制 ----------------
    def _check_screen_permission(self):
        """检查『屏幕录制』权限；未授予则弹明确警告（解释遮挡即失效成因与修复办法）。"""
        try:
            st = screen_capture_access_status()
        except Exception:  # pragma: no cover
            st = "unknown"
        if st == "denied":
            QMessageBox.warning(
                self, "需要『屏幕录制』权限",
                "检测到『屏幕录制』权限未授予。\n\n"
                "在 macOS 上，未授权时截图里其它 App 的窗口会被系统涂成纯色/空白——"
                "监控区域一旦被别的窗口遮挡，那块画面就『消失』，模板永远匹配不上，"
                "表现为你遇到的『有窗口遮挡之后检测失效』。\n\n"
                "解决办法：\n"
                "1) 打开『系统设置 → 隐私与安全性 → 屏幕录制』\n"
                "2) 找到并开启本 App（ScreenDetector）\n"
                "3) 完全退出并重新打开本 App 使权限生效\n\n"
                "授权后，被窗口遮挡的区域也能正常抓取，检测即可恢复。"
                "（注意：目标被完全遮住时仍无法检测，属物理限制。）")

    def _toggle_detect(self):
        if self.detect_timer.isActive():
            self._stop_detect()
            return
        if not self.region:
            QMessageBox.information(self, "提示", "请先设置监控区域（框选）。")
            return
        # 开始检测前再校验一次屏幕录制权限：未授权时遮挡区域抓不到，检测必然失效
        try:
            if screen_capture_access_status() == "denied":
                self._check_screen_permission()
                return
        except Exception:  # pragma: no cover
            pass
        mode = self._match_mode()
        targets = self._text_targets()
        if mode == 0 and not self.detector.templates:
            QMessageBox.information(self, "提示", "『仅图案』模式需要先添加至少一个检测图案。")
            return
        if mode == 1 and not targets:
            QMessageBox.information(self, "提示", "『仅文字』模式需要先输入『目标文字』。\n"
                                     "（在左侧『目标』框输入要检测的文字，多个用逗号隔开。）")
            return
        if mode == 2 and (not self.detector.templates or not targets):
            QMessageBox.information(self, "提示", "『文字优先·图案兜底』模式需要同时：添加检测图案 + 输入目标文字。")
            return
        scales = DEFAULT_SCALES if self.chk_ms.isChecked() else (1.0,)
        self.detector.set_scales(scales)
        # 重置上一轮可能残留的响铃/暂停状态，确保新一轮从干净状态开始
        self.alerting = False
        self.paused = False
        self._stop_resume_timer()
        self._prev_matched = False
        # 重置文字去抖状态，避免上一轮残留的"已确认命中"影响本轮
        self._text_hit_streak = 0
        self._text_miss_streak = 0
        self._text_detected = False
        self._occlusion_since = None  # 新一轮从"未遮挡"开始计时
        self.detect_timer.setInterval(max(100, self.spin_interval.value() * 1000))
        self.preview_timer.setInterval(200)  # 预览 200ms(5fps)：足够流畅且抓屏开销减半
        self.detect_timer.start()
        self.preview_timer.start()
        self._detecting = True
        self.btn_start.setText("⏹ 停止检测")
        self.btn_start.setStyleSheet("background:#ff3b30;color:#fff;padding:8px;font-weight:bold;")
        self._set_ring_button("idle")   # 检测进行中但还未响铃：按钮置灰
        self.lbl_status.setText("监控中…")
        # 立即把检测状态从"已停止"切回"监控中"，避免重启后状态卡在已停止
        self._set_hit_status("监控中", "#1565c0", "正在监控，等待图案出现…")

    def _stop_detect(self):
        self._detecting = False   # 先置否，丢弃 worker 残留的迟到 tick
        self.detect_timer.stop()
        # 注意：不停 preview_timer——停止检测后监控预览仍持续刷新，
        # 用户框选后可一直看到实时画面，无需重新点『开始检测』
        self._reset_ring()
        # 清空上一轮检测残留的框/文字，使停止后的预览为干净的纯实时画面（不带旧框）
        self._last_results = []
        self._last_recognized = None
        self._last_matched_texts = set()
        self.btn_start.setText("▶ 开始检测")
        self.btn_start.setStyleSheet("background:#0071e3;color:#fff;padding:8px;font-weight:bold;")
        self._set_ring_button("stopped")
        self.lbl_status.setText("已停止")
        self._set_hit_status("已停止", "#607d8b", "已停止检测（可重新点『开始检测』）。")

    # ---------------- 检测节拍（定时器驱动，持续监控） ----------------
    def _detect_step(self):
        """检测节拍：每『检测间隔』抓一帧，跑一次 OCR 文字检测 + 图案匹配，命中即响铃（持续监控）。

        遮挡自恢复：物理遮挡（窗口盖住区域 / 全屏切换 / 抓取暂不可用）时只跳过本帧、持续重试，
        绝不中断监控。整段包 try/except，确保任何单帧异常（匹配/OCR/绘制）都不会让 QTimer 死掉——
        这正是『有物理遮挡后整个检测停止、学习反馈也没了』的根因：异常冒泡到主线程槽会中断节拍。
        """
        if not self._detecting or not self.region:
            return
        try:
            x, y, w, h = self.region
            frame = self.cap.capture_points(x, y, w, h)
            if frame is None:
                self._on_occlusion()   # 标记遮挡 + 持续重试状态 + 起止日志（监控不中断）
                return
            self._on_capture_recovered()  # 若刚从遮挡恢复，记一条恢复日志
            mode = self._match_mode()
            need_text = (mode in (1, 2))
            need_pattern = (mode in (0, 2))
            results, debug = [], {}
            if need_pattern:
                try:
                    results, debug = self.detector.match_frame(frame, return_debug=True)
                except Exception as e:  # pragma: no cover
                    print("匹配异常(已跳过本帧):", repr(e))
                    results, debug = [], {}
            recognized = None
            targets = self._text_targets()
            if need_text and targets:
                try:
                    recognized = recognize_text(frame)
                except Exception as e:  # pragma: no cover
                    print("OCR 异常(本帧跳过):", repr(e))
                    recognized = None
            # 交给统一判定（文字 / 图案 / 兜底 + 响铃状态机）
            self._apply_detection(frame, results, debug,
                                  recognized=recognized, targets=targets,
                                  mode=mode, need_text=need_text)
        except Exception as e:  # pragma: no cover
            # 任何未预期异常都只跳过本帧，监控持续运行；不清空 _detecting、不停定时器
            print("检测节拍异常(已跳过本帧，监控继续):", repr(e))
            self.lbl_status.setText("⚠ 本帧检测异常，监控持续重试中…")

    def _on_occlusion(self):
        """监控区域抓取暂不可用（物理遮挡/全屏切换/权限问题）：保持监控持续重试，并记录起止。"""
        now = time.time()
        if self._occlusion_since is None:
            self._occlusion_since = now
            self._log("⚠ 监控区域抓取暂不可用（可能被窗口遮挡/全屏切换/权限问题），监控持续重试中…")
        elapsed = int(now - self._occlusion_since)
        self.lbl_status.setText(f"⚠ 区域被遮挡，监控持续重试中（已 {elapsed}s）")
        self._set_hit_status("遮挡·重试中", "#f9a825",
                             f"监控区域抓取暂不可用（可能被窗口遮挡），监控持续重试中，已 {elapsed}s。\n"
                             "遮挡解除后会自动恢复检测与运行日志，无需手动重启。")

    def _on_capture_recovered(self):
        """抓取从遮挡中恢复：清遮挡计时，并在『运行日志』里记一条恢复日志。"""
        if self._occlusion_since is not None:
            elapsed = int(time.time() - self._occlusion_since)
            self._occlusion_since = None
            self._log(f"✓ 遮挡解除，检测已恢复（遮挡持续约 {elapsed}s），运行日志继续。")

    def _preview_step(self):
        """预览节拍：只抓帧刷新画面（不跑 OCR、不响铃），保证监控画面实时流畅。

        CPU 优化（v6.24）：
          - 节拍由 100ms(10fps) 降到 200ms(5fps)——监控场景下足够流畅，抓屏与
            缩放/绘制开销直接减半。
          - 窗口最小化或不可见时直接跳过抓屏（return），避免后台空转持续抓屏占用 CPU。
        同样整段包 try/except：遮挡或任何单帧异常都只跳过本帧，绝不中断预览节拍。
        """
        if not self.region:
            return
        # 最小化/不可见时不抓屏：既省 CPU，也避免取到被遮挡的退化画面
        if self.isMinimized() or not self.isVisible():
            return
        try:
            x, y, w, h = self.region
            frame = self.cap.capture_points(x, y, w, h)
            if frame is None:
                return  # 遮挡期间预览不刷新即可，监控节拍负责重试与状态
            # 复用最近一次检测识别到的文字框 / 图案框，避免两次检测之间框消失
            self._update_preview(frame, self._last_results, {},
                                 text_items=self._last_recognized,
                                 matched_texts=self._last_matched_texts)
        except Exception as e:  # pragma: no cover
            print("预览节拍异常(已跳过本帧):", repr(e))

    def _apply_detection(self, frame, results, debug, recognized=None, targets=None, mode=None, need_text=None):
        if not self._detecting:
            return  # 已停止检测，丢弃残留 tick，避免覆盖"已停止"状态
        if results is None:
            self.lbl_status.setText("⚠ 抓取失败（可能未授予屏幕录制权限）")
            self._set_hit_status("抓取失败", "#b71c1c",
                                 "无法抓取监控区域，请确认已授予『屏幕录制』权限。")
            self._last_debug = {}
            return
        self._last_debug = debug or {}
        if mode is None:
            mode = self._match_mode()
        if need_text is None:
            need_text = (mode in (1, 2))
        pure_text = (mode == 1)
        pattern_matched = bool(results)
        matched_list = []
        # —— 文字命中（OCR 已在检测节拍 _detect_step 内按『检测间隔』跑过，recognized 由参数传入）——
        if targets is None:
            targets = self._text_targets()
        texts = [r["text"] for r in (recognized or [])]
        if need_text and targets:
            raw_matched, mlist = text_matches(texts, targets, self._text_mode_key())
            matched_list = mlist
            if raw_matched:
                self._text_hit_streak += 1
                self._text_miss_streak = 0
            else:
                self._text_miss_streak += 1
                self._text_hit_streak = 0
            if self._text_hit_streak >= TEXT_HIT_CONFIRM:
                self._text_detected = True
            if self._text_miss_streak >= TEXT_CLEAR_CONFIRM:
                self._text_detected = False
            text_detected = self._text_detected
        else:
            text_detected = False
        text_items = recognized if need_text else None
        matched_texts = set(matched_list)
        # 缓存最近一次识别结果，供预览节拍绘制文字框 / 图案框使用
        self._last_recognized = recognized
        self._last_matched_texts = matched_texts
        self._last_results = results
        # —— 综合命中判定 ——
        if mode == 0:
            matched = pattern_matched
        elif mode == 1:
            matched = text_detected
        else:  # mode == 2：文字优先，文字检测不到时再用图案兜底（文字 OR 图案）
            matched = text_detected or pattern_matched
        # 预览：仅文字模式不画模板框；其余画模板框；文字框始终按 need_text 画
        results_for_preview = [] if pure_text else results
        if self.region:
            self._update_preview(frame, results_for_preview, debug,
                                 text_items=text_items, matched_texts=matched_texts)
        # —— 工具内实时命中状态（每帧更新）——
        now = __import__("datetime").datetime.now().strftime("%H:%M:%S")
        if pure_text:
            if targets:
                if matched:
                    self._set_hit_status("🟢 文字命中", "#2e7d32",
                                         f"命中文字：{', '.join(matched_list)}\n时间 {now}")
                else:
                    sample = (f"如：{', '.join(texts[:5])}" if texts else "暂未识别到文字")
                    self._set_hit_status("⚪ 未命中", "#607d8b",
                                         f"监控中（文字模式）：识别到 {len(texts)} 段文字；{sample}（{now}）")
                    if texts and targets:
                        import time
                        if time.time() - self._last_text_diag_log > 5:
                            self._last_text_diag_log = time.time()
                            self._log(f"文字未命中诊断：目标 {targets}；识别到 {texts[:10]}")
            else:
                self._set_hit_status("⚪ 未设置目标", "#607d8b",
                                     "『仅文字』模式已选，但『目标文字』为空——请在左侧输入要检测的文字。")
        elif mode == 2:
            # 文字优先·图案兜底：分别展示图案与文字状态，并指出本次由谁触发
            p = "✅图案" if pattern_matched else "⚪图案"
            t = "✅文字" if text_detected else "⚪文字"
            if matched:
                if text_detected and pattern_matched:
                    self._set_hit_status("🟢 命中（文字+图案）", "#2e7d32",
                                         f"同时命中：{t} + {p}\n文字：{', '.join(matched_list) or '无'}\n时间 {now}")
                elif text_detected:
                    self._set_hit_status("🟢 命中（文字）", "#2e7d32",
                                         f"文字命中：{', '.join(matched_list)}\n（图案未命中，但文字优先已触发）\n时间 {now}")
                else:  # 文字未命中，图案兜底命中
                    self._set_hit_status("🟢 命中（图案兜底）", "#2e7d32",
                                         f"文字未检测到目标，由图案兜底命中\n{p}\n时间 {now}")
            else:
                self._set_hit_status("⚪ 未命中", "#607d8b",
                                     f"监控中（文字优先·图案兜底）：{t} + {p}（文字检测到 或 图案命中即响铃，{now}）")
        else:  # mode == 0 仅图案
            if results:
                names = "、".join(f"{r.name}({r.score:.2f})" for r in results[:3])
                self._set_hit_status("🟢 命中（图案）", "#2e7d32",
                                     f"图案命中：{names}\n时间 {now}")
            else:
                best_info = self._best_debug_info(debug)
                if best_info:
                    detail = (f"监控中（仅图案）：未命中；最接近 "
                              f"{best_info['name']} {best_info['best_score']:.2f}/"
                              f"{best_info['threshold']:.2f}（{now}）")
                else:
                    detail = f"监控中（仅图案）：未命中（{now}）"
                self._set_hit_status("⚪ 未命中", "#607d8b", detail)
        # —— 响铃状态机（图案/文字统一）：暂停响铃 + 可切换的『手动/自动恢复』 ——
        # 响铃恢复方式（ring_resume_mode）：
        #   0 手动恢复：暂停后必须手动点『启动响铃』才恢复。
        #   1 新命中自动恢复：暂停只压制"当前这一轮持续命中"；当目标消失后再出现（新一轮命中 = 上升沿）自动解除暂停并重新响铃，不漏新事件。
        #   2 自定义秒数后自动解除：暂停起 N 秒倒计时到点自动解除暂停（定时器在 _pause_ring 启动，N 由用户自定义）；若届时仍有目标命中会立即重新响铃。
        #   3 命中跟随：响铃严格跟随命中状态——命中即响、消失即停；手动暂停后下一次命中自动恢复。
        # 命中跟随模式：命中时强制解除暂停，确保"命中关闭静音、开始播放提示音"
        if self.ring_resume_mode == 3 and matched:
            self.paused = False
            self._stop_resume_timer()
        rising = matched and not self._prev_matched
        if matched and not self.alerting:
            # 上升沿 / 暂停中的同一轮命中：开始响铃（按命中来源确定对应提示音）
            if self.paused and self.ring_resume_mode == 1 and rising:
                # 自动恢复：新一轮命中自动解除暂停
                self.paused = False
            if not self.paused:
                self.alerting = True
                ring_name, ring_path = self._resolve_ring_sound(mode, results, matched_list)
                self._start_beep(ring_name, ring_path)
                if pure_text:
                    self.lbl_status.setText(f"🔔 响铃中：命中文字 {', '.join(matched_list)}")
                    self._set_ring_button("ringing")
                    self._log(f"触发响铃(文字): {', '.join(matched_list)}")
                elif results:
                    # 图案命中（仅图案 或 文字+图案且图案命中）：弹反馈卡
                    best = results[0]
                    self.current_feedback = (best.tid, best.score, best.crop)
                    self._show_feedback_card(best.name, best.score, best.crop)
                    self._set_ring_button("ringing")   # 响铃中：按钮变为可点的橙色"停止响铃"
                    suffix = "" if mode == 0 else f" + 文字 {', '.join(matched_list)}"
                    source_tag = "图案" if mode == 0 else "图案兜底"
                    self.lbl_status.setText(f"🔔 响铃中：{source_tag}命中 {best.name}（匹配度 {best.score:.2f}）{suffix}")
                    self._log(f"触发响铃({source_tag}): {best.name} ({best.score:.2f}){(' + 文字 ' + str(matched_list)) if matched_list else ''}")
                else:
                    # 文字+图案模式、文字命中但图案未命中：文字优先触发，无图案反馈卡
                    self.lbl_status.setText(f"🔔 响铃中：命中文字 {', '.join(matched_list)}（图案兜底未触发）")
                    self._set_ring_button("ringing")
                    self._log(f"触发响铃(文字优先): {', '.join(matched_list)}")
            else:
                # 处于暂停期：只显示命中状态但不响铃、不弹反馈、不重复记录
                if self.ring_resume_mode == 1:
                    auto_note = "（自动恢复模式下，目标消失再出现将自动解除暂停并响铃）"
                elif self.ring_resume_mode == 2:
                    auto_note = f"（暂停 {self.ring_resume_seconds} 秒后自动解除暂停，或点击『启动响铃』立即恢复）"
                else:
                    auto_note = "（点击『启动响铃』可继续）"
                if pure_text or not results:
                    self._set_hit_status("🟡 命中（已暂停）", "#1565c0",
                                         f"检测到文字 {', '.join(matched_list)}，但已暂停响铃。{auto_note}")
                else:
                    best = results[0]
                    source_tag = "图案" if mode == 0 else "图案兜底"
                    self._set_hit_status("🟡 命中（已暂停）", "#1565c0",
                                         f"{source_tag}命中 {best.name}（{best.score:.2f}）+ 文字，但已暂停响铃。{auto_note}")
        elif self.alerting and not matched:
            # 下降沿：目标消失 -> 自动停铃；自动恢复模式下同时清除暂停，等待下一轮新命中
            self.sound.stop()
            if self.beep_timer:
                self.beep_timer.stop()
            self.alerting = False
            if self.ring_resume_mode == 1:
                self.paused = False
            self._set_ring_button("paused" if self.paused else "idle")
        elif self.alerting and matched:
            # 持续检测到：保持响铃（beep 由定时器负责）
            pass
        elif not matched:
            # 持续未命中：若仍在响铃则停铃；自动恢复模式下清除暂停（手动模式保持暂停，等待手动恢复）
            if self.alerting:
                self.sound.stop()
                if self.beep_timer:
                    self.beep_timer.stop()
                self.alerting = False
            if self.ring_resume_mode == 1 and self.paused:
                self.paused = False
            self._set_ring_button("paused" if self.paused else "idle")
        self._prev_matched = matched

    def _start_beep(self, sound_name=None, custom_path=""):
        """开始响铃。sound_name/custom_path 指定本次响铃的声音（不同图案/文字可不同）；
        省略则用全局默认声音。每次播放新声音前会先停掉上一个，避免声音叠加。"""
        if sound_name is None:
            sound_name, custom_path = self._get_sound_args()
        self._ring_sound = sound_name
        self._ring_custom = custom_path
        if self.beep_timer is None:
            self.beep_timer = QTimer(self)
            self.beep_timer.timeout.connect(self._beep)
        self.beep_timer.start(max(300, self.spin_cooldown.value() * 1000))
        self._beep()

    def _beep(self):
        self.sound.play(self._ring_sound, custom_path=self._ring_custom)

    # ---------------- 响铃声音解析（不同图案/文字不同提示音）----------------
    def _sound_for_tid(self, tid):
        """返回某图案命中时应播放的声音 (name, custom_path)。空 sound=用全局。"""
        t = self.detector.get(tid)
        s = t.sound if t and getattr(t, "sound", "") else ""
        if s:
            return s, ""
        return self._get_sound_args()

    def _text_sound_for(self, matched_list):
        """返回命中文字对应的提示音：优先取第一个命中文字单独指定的声音；未指定则回退最下方全局默认。"""
        for t in (matched_list or []):
            s = self._text_sounds.get(t, "")
            if s:
                return s, ""
        return self._get_sound_args()

    def _resolve_ring_sound(self, mode, results, matched_list=None):
        """根据命中来源决定本次响铃声音：优先用对应图案/文字各自的提示音，全部默认则回退最下方全局声音。"""
        if mode == 0 and results:
            return self._sound_for_tid(results[0].tid)
        if mode == 1:
            return self._text_sound_for(matched_list)
        if mode == 2:
            if results:
                return self._sound_for_tid(results[0].tid)
            return self._text_sound_for(matched_list)
        return self._get_sound_args()

    def _pause_ring(self):
        """暂停响铃：立即消音但保持监控。按当前恢复模式决定后续：手动需点按钮；新命中自动靠下降沿；N秒自动靠定时器。"""
        self.sound.stop()          # 立即消音（杀掉正在播放/尾音）
        if self.beep_timer:
            self.beep_timer.stop()
        self.alerting = False
        self.paused = True
        self.feedback_card.hide()
        self._set_ring_button("paused")   # 按钮变为可点的蓝色"启动响铃"
        if self.ring_resume_mode == 1:
            note = "目标消失再出现将自动恢复响铃"
        elif self.ring_resume_mode == 2:
            note = f"{self.ring_resume_seconds} 秒后自动解除暂停并恢复响铃"
            self._start_resume_timer()
        elif self.ring_resume_mode == 3:
            note = "下一次命中将自动恢复响铃"
        else:
            note = "需点击『启动响铃』手动恢复"
        self._set_hit_status("⏸ 已暂停响铃", "#1565c0",
                             f"已暂停响铃；{note}，或点击『启动响铃』立即恢复。")
        mode_name = {0: "手动恢复", 1: "新命中自动恢复", 2: f"{self.ring_resume_seconds}秒后自动恢复", 3: "命中跟随"}[self.ring_resume_mode]
        self._log("用户暂停响铃 -> " + mode_name)

    def _reset_ring(self):
        """彻底重置响铃/暂停状态（停止检测或重新开始时调用）。"""
        self.sound.stop()
        if self.beep_timer:
            self.beep_timer.stop()
        self._stop_resume_timer()
        self.alerting = False
        self.paused = False
        self._prev_matched = False
        self.feedback_card.hide()
        self.current_feedback = None

    def _resume_ring(self):
        """手动启动响铃：解除暂停，后续命中将正常响铃（下一次检测节拍即生效）。"""
        self._stop_resume_timer()
        self.paused = False
        self._set_ring_button("idle")
        self._set_hit_status("监控中", "#1565c0", "已启动响铃；等待命中…")
        self._log("用户启动响铃")

    def _on_ring_button_clicked(self):
        """暂停/启动响铃按钮：未暂停 -> 暂停；已暂停 -> 启动。形成清晰开关。"""
        if self.paused:
            self._resume_ring()
        else:
            self._pause_ring()

    def _start_resume_timer(self):
        """启动 N 秒倒计时，到点自动解除暂停（仅 mode==2 使用，N 由 ring_resume_seconds 决定）。"""
        self._stop_resume_timer()
        self._resume_timer = QTimer(self)
        self._resume_timer.setSingleShot(True)
        self._resume_timer.timeout.connect(self._on_resume_timeout)
        self._resume_timer.start(max(1000, self.ring_resume_seconds * 1000))

    def _stop_resume_timer(self):
        """停止并销毁 10 秒自动解除暂停定时器（若存在）。"""
        if self._resume_timer is not None:
            try:
                self._resume_timer.stop()
            except Exception:
                pass
            self._resume_timer = None

    def _on_resume_timeout(self):
        """N 秒到点：自动解除暂停（若此刻有目标命中，下一帧检测会立即重新响铃）。"""
        self._resume_timer = None
        if not self.paused:
            return
        self.paused = False
        self._set_ring_button("idle")
        self._set_hit_status("监控中", "#1565c0", f"{self.ring_resume_seconds} 秒已到，已自动解除暂停；等待命中…")
        self._log(f"{self.ring_resume_seconds} 秒自动解除暂停")

    def _on_ring_mode_changed(self, index: int):
        """『暂停后恢复方式』下拉：切换恢复模式并即时持久化；同步启用/禁用秒数输入。"""
        self.ring_resume_mode = index
        self.spin_ring_seconds.setEnabled(index == 2)
        # 切到『新命中自动恢复』且当前处于『无命中却仍暂停』的陈旧状态，顺手清除，避免一直静音
        if index == 1 and not self._prev_matched and self.paused:
            self.paused = False
            self._set_ring_button("idle")
        # 离开『自定义秒数后恢复』模式时，若正挂着倒计时则取消（避免后续误触发）
        if index != 2:
            self._stop_resume_timer()
        # 切到『自定义秒数后恢复』且当前已处于暂停，立即启动倒计时（秒数可能已变）
        if index == 2 and self.paused:
            self._start_resume_timer()
        self._save_config()

    def _on_ring_seconds_changed(self, value: int):
        """自定义秒数输入：更新值并即时持久化；若当前正处于 mode==2 的暂停倒计时，重启倒计时以应用新秒数。"""
        self.ring_resume_seconds = max(1, min(600, int(value)))
        if self.ring_resume_mode == 2 and self.paused:
            self._start_resume_timer()
            self._set_hit_status("⏸ 已暂停响铃", "#1565c0",
                                 f"已暂停响铃；{self.ring_resume_seconds} 秒后自动解除暂停并恢复响铃，或点击『启动响铃』立即恢复。")
        self._save_config()

    def _show_feedback_card(self, name, score, crop):
        """在主窗口右侧显示命中确认卡片（替代独立弹窗）。"""
        self.fb_title.setText(f"检测到：{name}")
        self.fb_info.setText(f"匹配度 {score:.2f}\n请确认是否成功命中")
        if crop is not None and crop.size:
            self.fb_thumb.setPixmap(numpy_to_pixmap(crop, 120, 80))
        else:
            self.fb_thumb.setText("无缩略图")
        self.feedback_card.show()

    # ---------------- 反馈学习 ----------------
    def _on_feedback(self, is_hit: bool):
        if not self.current_feedback:
            return
        tid, score, crop = self.current_feedback
        info = self.detector.record_feedback(tid, is_hit, score, crop)
        self.current_feedback = None
        # 无论『命中』还是『误报』，用户已确认本次检测结果，立即暂停响铃；
        # 暂停而非彻底停止——点击『启动响铃』后可继续，后续命中将重新响铃。
        self._pause_ring()
        self._set_hit_status("⏸ 已记录反馈·已暂停响铃", "#1565c0",
                             f"反馈已记录（{'命中✓' if is_hit else '误报✗'}），响铃已暂停；点击『启动响铃』可继续。")
        self.feedback_card.hide()
        self.fb_title.setText("检测到图案")
        self.fb_info.setText("匹配度 -\n请点击下方按钮确认")
        self.fb_thumb.setText("无缩略图")
        self.fb_thumb.setPixmap(QPixmap())
        name = self.detector.get(tid)
        n = name.name if name else tid
        self._log(f"反馈[{n}]: {'命中✓' if is_hit else '误报✗'} -> 有效阈值 {info.get('effective_threshold',0):.2f}")
        self._refresh_tpl_list()
        self._save_config()

    # ---------------- 预览 ----------------
    def _update_preview(self, frame, results, debug=None, text_items=None, matched_texts=None):
        if frame is None:
            return
        disp = frame.copy()
        # 诊断：整帧接近全黑 -> 提示抓取异常（区域越界/权限）
        if float(np.mean(frame)) < 6:
            cv2.putText(disp, "[抓取为黑屏]", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        # 画出命中框
        for r in results:
            cv2.rectangle(disp, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 2)
            cv2.putText(disp, f"{r.name} {r.score:.2f}", (r.x, max(0, r.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # 画出文字识别框（绿色：普通识别；橙色：命中目标）
        if text_items:
            for t in text_items:
                hit = matched_texts and _normalize_text(t["text"]) in matched_texts
                color = (0, 165, 255) if hit else (0, 255, 0)  # 橙 / 绿
                cv2.rectangle(disp, (t["x"], t["y"]),
                               (t["x"] + t["w"], t["y"] + t["h"]), color, 2)
                cv2.putText(disp, t["text"], (t["x"], max(0, t["y"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        # 若未命中但接近阈值，把最佳匹配位置用虚线/灰色框标出，帮助用户定位
        if not results and debug:
            best = self._best_debug_info(debug)
            if best and best['best_score'] >= 0.55:
                x, y = best['loc']
                # 用 scale 反推当前模板在画面中的近似尺寸
                item = self.tpl_list.currentItem()
                tid = item.data(Qt.UserRole) if item else None
                t = self.detector.get(tid) if tid else None
                if t and best['tid'] == tid:
                    tw = int(round(t.active_image.shape[1] * best['scale']))
                    th = int(round(t.active_image.shape[0] * best['scale']))
                    cv2.rectangle(disp, (x, y), (x + tw, y + th), (0, 255, 255), 2)
                    cv2.putText(disp, f"~{best['name']} {best['score']:.2f}",
                                (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)
        self._render_preview(disp)

    def _render_preview(self, disp):
        """把标注后的画面渲染到 QLabel，支持铺满显示 + 用户缩放/平移。

        默认按 QLabel 尺寸 cover 铺满，让框选画面占满整个预览区；
        用户可通过 ＋/－/方向键 缩放平移查看细节，点击重置恢复默认铺满视图。
        缩放以画面中心为锚点：缩小时画面居中于 QLabel，四周以黑边补齐。
        """
        label_w = self.preview.width()
        label_h = self.preview.height()
        if disp is None or label_w <= 0 or label_h <= 0:
            return
        h, w = disp.shape[:2]
        if w == 0 or h == 0:
            return
        # 1) cover 模式基础缩放：让画面至少铺满 QLabel，但不限制放大比例
        base_scale = max(label_w / w, label_h / h)
        total_scale = base_scale * self._preview_scale
        sw = int(round(w * total_scale))
        sh = int(round(h * total_scale))
        if sw < 1 or sh < 1:
            return
        scaled = cv2.resize(disp, (sw, sh), interpolation=cv2.INTER_LINEAR)
        # 2) 计算居中裁剪/放置位置（scaled 图像坐标系中，裁剪出 QLabel 大小区域的左上角）
        base_x = (sw - label_w) // 2
        base_y = (sh - label_h) // 2
        sx = base_x + self._preview_offset.x()
        sy = base_y + self._preview_offset.y()
        # 限制用户偏移，避免画面完全移出可视区
        if sw <= label_w:
            min_sx = -(label_w - sw) // 2
            max_sx = 0
        else:
            min_sx = 0
            max_sx = sw - label_w
        if sh <= label_h:
            min_sy = -(label_h - sh) // 2
            max_sy = 0
        else:
            min_sy = 0
            max_sy = sh - label_h
        sx = max(min_sx, min(sx, max_sx))
        sy = max(min_sy, min(sy, max_sy))
        self._preview_offset = QPoint(sx - base_x, sy - base_y)
        # 3) 以黑底为画布，把 scaled 上对应区域贴到 QLabel（超出部分以黑边补齐）
        pad = np.zeros((label_h, label_w, 3), dtype=np.uint8)
        x1 = max(0, sx)
        y1 = max(0, sy)
        x2 = min(sw, sx + label_w)
        y2 = min(sh, sy + label_h)
        dx1 = max(0, -sx)
        dy1 = max(0, -sy)
        dx2 = dx1 + (x2 - x1)
        dy2 = dy1 + (y2 - y1)
        if x2 > x1 and y2 > y1:
            pad[dy1:dy2, dx1:dx2] = scaled[y1:y2, x1:x2]
        self._preview_base_pixmap = numpy_to_pixmap(pad)
        self.preview.setPixmap(self._preview_base_pixmap)

    def _reset_preview_view(self):
        self._preview_scale = 1.0
        self._preview_offset = QPoint(0, 0)
        self._preview_base_pixmap = None

    def _zoom_preview(self, factor):
        self._preview_scale = max(0.1, min(5.0, self._preview_scale * factor))

    def _pan_preview(self, dx, dy):
        off = self._preview_offset
        self._preview_offset = QPoint(off.x() + dx, off.y() + dy)

    def _best_debug_info(self, debug):
        """返回所有启用模板中最佳匹配得分项，用于 UI 诊断。"""
        if not debug:
            return None
        best = None
        for tid, info in debug.items():
            # 注意调试字典键是 best_score（不是 score）
            sc = info.get('best_score', 0.0)
            if best is None or sc > best.get('best_score', -1.0):
                best = {'tid': tid, **info}
        return best

    def _test_match_template(self):
        """手动测试当前选中模板在当前监控区域的匹配情况。"""
        item = self.tpl_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先选中一个检测图案。")
            return
        tid = item.data(Qt.UserRole)
        t = self.detector.get(tid)
        if not t:
            return
        if not self.region:
            QMessageBox.information(self, "提示", "请先设置监控区域。")
            return
        x, y, w, h = self.region
        frame = self.cap.capture_points(x, y, w, h)
        if frame is None:
            QMessageBox.warning(self, "失败", "抓取监控区域失败，请确认已授予屏幕录制权限。")
            return
        # 使用完整多尺度
        scales = DEFAULT_SCALES if self.chk_ms.isChecked() else (1.0,)
        self.detector.set_scales(scales)
        debug = self.detector.debug_match_scores(frame, only_tids=[tid])
        info = debug.get(tid, {})
        score = info.get('best_score', -1.0)
        threshold = info.get('threshold', t.base_threshold)
        loc = info.get('loc', (0, 0))
        scale = info.get('scale', 1.0)
        hit = "✓ 命中" if score >= threshold else "✗ 未过线"
        self._log(f"测试匹配[{t.name}]: 得分 {score:.3f} / 阈值 {threshold:.2f} @({loc[0]},{loc[1]}) 尺度{scale:.2f} → {hit}")
        self.det_live.setText(f"当前最佳匹配度: {score:.2f} {'✓' if score >= threshold else '✗'}")
        # 在预览中标出测试位置
        disp = frame.copy()
        tw = int(round(t.active_image.shape[1] * scale))
        th = int(round(t.active_image.shape[0] * scale))
        lx, ly = loc
        color = (0, 255, 0) if score >= threshold else (0, 255, 255)
        cv2.rectangle(disp, (lx, ly), (lx + tw, ly + th), color, 2)
        cv2.putText(disp, f"{t.name} {score:.2f}", (lx, max(0, ly - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self._render_preview(disp)
        if score < threshold:
            QMessageBox.information(
                self, "测试结果",
                f"{t.name} 未命中。\n\n"
                f"最佳得分：{score:.3f}\n"
                f"有效阈值：{threshold:.2f}\n\n"
                f"建议：\n"
                f"1) 若得分接近阈值，可调低『基准阈值』；\n"
                f"2) 若得分很低（<0.6），说明画面中的目标与模板视觉样式差异较大，"
                f"请重新从监控区域中框选目标作为模板。")
        else:
            QMessageBox.information(self, "测试结果", f"{t.name} 命中！\n得分 {score:.3f} ≥ 阈值 {threshold:.2f}")

    def _save_template_image(self):
        """把当前选中模板的图片（框选/载入的原始图案）导出为 PNG 保存到任意位置。

        此前模板图片仅在工具内部持久化（~/Library/Application Support/ScreenPatternDetector），
        没有面向用户的存档入口；本按钮让用户可以把框选/载入的图案另存为独立图片文件，
        便于复用、归档或分享。
        """
        item = self.tpl_list.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先选中一个检测图案，再点『保存图案图片』。")
            return
        tid = item.data(Qt.UserRole)
        t = self.detector.get(tid)
        if not t:
            return
        img = t.image
        if img is None or not img.size:
            QMessageBox.warning(self, "失败", "该模板没有可保存的图片数据。")
            return
        # 默认文件名：用模板名（清理非法字符），默认存到桌面
        safe = re.sub(r'[\\/:*?"<>|]', '_', t.name) or "pattern"
        default = os.path.join(os.path.expanduser("~/Desktop"), f"{safe}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图案图片", default,
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;所有文件 (*)")
        if not path:
            return
        # 规范化扩展名：未带后缀时默认补 .png
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            path += ".png"
        ok = cv2.imwrite(path, img)
        if ok:
            self._show_msg("保存成功",
                f"已将『{t.name}』的图案图片保存到：\n{path}")
            self._log(f"已保存模板图片[{t.name}]: {path}")
        else:
            self._show_msg("保存失败", f"无法写入文件：\n{path}", icon=QMessageBox.Warning)

    # ---------------- 文字检测 ----------------
    def _match_mode(self):
        """0=仅图案 1=仅文字 2=文字优先·图案兜底（文字 OR 图案）。"""
        cm = getattr(self, "combo_match_mode", None)
        if cm is None:
            return 0
        return cm.currentIndex()

    def _text_targets(self):
        import re
        raw = self.le_target_text.text().strip()
        if not raw:
            return []
        parts = re.split(r"[,，、\s]+", raw)
        return [p.strip() for p in parts if p.strip()]

    def _text_mode_key(self):
        return ["any", "all", "equal"][self.combo_text_mode.currentIndex()]

    def _test_text_detect(self):
        """手动测试：对当前监控区域跑一次 OCR，展示识别结果与是否命中目标。"""
        if not vision_available():
            self._show_msg("文字检测不可用",
                "当前环境未加载 pyobjc-framework-Vision，无法使用原生 OCR。\n"
                "请安装该框架后重试（pip install pyobjc-framework-Vision）。",
                icon=QMessageBox.Warning)
            return
        if not self.region:
            self._show_msg("提示", "请先设置监控区域（框选）。")
            return
        x, y, w, h = self.region
        frame = self.cap.capture_points(x, y, w, h)
        if frame is None:
            self._show_msg("失败", "抓取监控区域失败，请确认已授予屏幕录制权限。",
                           icon=QMessageBox.Warning)
            return
        recognized = recognize_text(frame)
        texts = [r["text"] for r in recognized]
        targets = self._text_targets()
        # 预览中画出识别框
        disp = frame.copy()
        for t in recognized:
            cv2.rectangle(disp, (t["x"], t["y"]), (t["x"] + t["w"], t["y"] + t["h"]), (0, 255, 0), 2)
            cv2.putText(disp, t["text"], (t["x"], max(0, t["y"] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        self._render_preview(disp)
        if not targets:
            self._show_msg("识别结果",
                f"识别到 {len(texts)} 段文字：\n"
                + ("、".join(texts[:20]) if texts else "（无）")
                + "\n\n请在『目标文字』中输入要检测的内容后再开始监控。")
            return
        hit, matched = text_matches(texts, targets, self._text_mode_key())
        self._show_msg("测试结果",
            f"识别到 {len(texts)} 段文字：{', '.join(texts[:20]) if texts else '（无）'}\n\n"
            f"目标：{', '.join(targets)}\n匹配方式：{self.combo_text_mode.currentText()}\n"
            f"结果：{'✓ 命中 ' + ', '.join(matched) if hit else '✗ 未命中'}")

    # ---------------- 其它 ----------------
    def _get_sound_args(self):
        """返回当前选择的 (sound_name, custom_path)。"""
        name = self.snd_combo.currentText()
        if name == "自定义…":
            return "Ping", self._custom_sound_path
        return name, ""

    def _browse_custom_sound(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择提示音文件", "",
            "音频文件 (*.aiff *.wav *.caf *.mp3 *.m4a);;所有文件 (*)")
        if path and os.path.isfile(path):
            self._custom_sound_path = path
            self.snd_combo.setCurrentText("自定义…")
            self.lbl_status.setText(f"已选择自定义提示音：{os.path.basename(path)}")
            self._save_config()

    def _on_sound_selection_changed(self):
        name = self.snd_combo.currentText()
        if name != "自定义…":
            self._custom_sound_path = ""
        self._save_config()

    def _test_sound(self):
        name, path = self._get_sound_args()
        self.sound.play(name, custom_path=path)

    def _log(self, msg):
        from datetime import datetime
        self.log.insertItem(0, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        if self.log.count() > 200:
            self.log.takeItem(self.log.count() - 1)

    def closeEvent(self, e):
        self._stop_detect()
        self._save_config()
        e.accept()


def main():
    app = QApplication([])
    app.setApplicationName("框选屏幕检测工具")
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
