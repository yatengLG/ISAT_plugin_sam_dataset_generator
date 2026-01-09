# -*- coding: utf-8 -*-
# @Author  : LG

import os
import random

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image

from ISAT.widgets.plugin_base import PluginBase
from ISAT.widgets.polygon import Polygon


class SamDatasetGeneratorPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.auto_segment_predictor = None

    def init_plugin(self, mainwindow):
        self.mainwindow = mainwindow
        self.init_ui()
        self.auto_segment_predictor = None

    def enable_plugin(self):
        self.mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)
        self.dock.show()
        self.init_auto_segment_predictor()
        self.enabled = True

    def disable_plugin(self):
        self.mainwindow.removeDockWidget(self.dock)
        self.enabled = False

    def get_plugin_author(self) -> str:
        try:
            from ISAT_plugin_sam_dataset_generator import __author__
        except Exception as e:
            __author__ = "unknown"
        return __author__

    def get_plugin_version(self) -> str:
        try:
            from ISAT_plugin_sam_dataset_generator import __version__
        except Exception as e:
            __version__ = "unknown"
        return __version__

    def get_plugin_description(self) -> str:
        try:
            from ISAT_plugin_sam_dataset_generator import __description__
        except Exception as e:
            __description__ = "unknown"
        return __description__

    def init_ui(self):
        self.dock = QtWidgets.QDockWidget(self.mainwindow)
        self.dock.setWindowTitle('Sam Dataset Generator')

        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.PushButton_predict = QtWidgets.QPushButton("Predict")
        self.PushButton_predict.clicked.connect(self.predict)
        main_layout.addWidget(self.PushButton_predict)

        self.CheckBox_random_category = QtWidgets.QCheckBox("random category")
        self.CheckBox_random_category.setChecked(True)
        main_layout.addWidget(self.CheckBox_random_category)
        
        # Points per side
        points_layout = QtWidgets.QHBoxLayout()
        points_label = QtWidgets.QLabel("Points per side:")
        self.SpinBox_points_per_side = QtWidgets.QSpinBox()
        self.SpinBox_points_per_side.setRange(8, 64)
        self.SpinBox_points_per_side.setValue(32)
        self.SpinBox_points_per_side.setToolTip("每边采样点数，值越大生成mask越多")
        points_layout.addWidget(points_label)
        points_layout.addWidget(self.SpinBox_points_per_side)
        main_layout.addLayout(points_layout)

        # Pred IoU threshold
        iou_layout = QtWidgets.QHBoxLayout()
        iou_label = QtWidgets.QLabel("Pred IoU thresh:")
        self.DoubleSpinBox_pred_iou_thresh = QtWidgets.QDoubleSpinBox()
        self.DoubleSpinBox_pred_iou_thresh.setRange(0.0, 1.0)
        self.DoubleSpinBox_pred_iou_thresh.setSingleStep(0.05)
        self.DoubleSpinBox_pred_iou_thresh.setValue(0.88)
        self.DoubleSpinBox_pred_iou_thresh.setToolTip("预测IoU阈值，值越高质量要求越严格")
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.DoubleSpinBox_pred_iou_thresh)
        main_layout.addLayout(iou_layout)

        # Stability score threshold
        stability_layout = QtWidgets.QHBoxLayout()
        stability_label = QtWidgets.QLabel("Stability thresh:")
        self.DoubleSpinBox_stability_thresh = QtWidgets.QDoubleSpinBox()
        self.DoubleSpinBox_stability_thresh.setRange(0.0, 1.0)
        self.DoubleSpinBox_stability_thresh.setSingleStep(0.05)
        self.DoubleSpinBox_stability_thresh.setValue(0.95)
        self.DoubleSpinBox_stability_thresh.setToolTip("稳定性分数阈值，控制mask质量")
        stability_layout.addWidget(stability_label)
        stability_layout.addWidget(self.DoubleSpinBox_stability_thresh)
        main_layout.addLayout(stability_layout)

        # Min mask region area
        area_layout = QtWidgets.QHBoxLayout()
        area_label = QtWidgets.QLabel("Min mask area:")
        self.SpinBox_min_mask_area = QtWidgets.QSpinBox()
        self.SpinBox_min_mask_area.setRange(0, 10000)
        self.SpinBox_min_mask_area.setSingleStep(100)
        self.SpinBox_min_mask_area.setValue(0)
        self.SpinBox_min_mask_area.setToolTip("最小mask区域面积（像素），过滤小碎片")
        area_layout.addWidget(area_label)
        area_layout.addWidget(self.SpinBox_min_mask_area)
        main_layout.addLayout(area_layout)
        
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMinimumSize(QtCore.QSize(0, 10))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setStyleSheet("QProgressBar {\n"
"            border: 1px solid #888783;\n"
"\n"
"            border-radius: 3px;\n"
"        }\n"
"QProgressBar::chunk {\n"
"            background-color: #74d65f;\n"
"            border-radius: 2px;\n"
"            width: 6px;\n"
"            margin: 1px;\n"
"        }")
        main_layout.addWidget(self.progressBar)

        self.dock.setWidget(main_widget)
        self.mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)

        if not self.enabled:
            self.disable_plugin()

    def init_auto_segment_predictor(self):
        if self.mainwindow.use_segment_anything:
            if self.mainwindow.segany.model_source == "mobile_sam":
                from ISAT.segment_any.mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
            elif self.mainwindow.segany.model_source == "edge_sam":
                from ISAT.segment_any.edge_sam.automatic_mask_generator import SamAutomaticMaskGenerator
            elif self.mainwindow.segany.model_source == "sam_hq":
                from ISAT.segment_any.segment_anything_hq.automatic_mask_generator import SamAutomaticMaskGenerator
            elif self.mainwindow.segany.model_source == "sam":
                from ISAT.segment_any.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
            elif self.mainwindow.segany.model_source == "sam2" or self.mainwindow.segany.model_source == "sam2.1":
                from ISAT.segment_any.sam2.automatic_mask_generator import \
                    SAM2AutomaticMaskGenerator as SamAutomaticMaskGenerator
            elif self.mainwindow.segany.model_source == "sam3":
                QtWidgets.QMessageBox.warning(self.mainwindow, "Warning", "SAM3 not support everything mode.")
                return
            elif self.mainwindow.segany.model_source == "sam_med2d":
                from ISAT.segment_any.segment_anything_med2d.automatic_mask_generator import SamAutomaticMaskGenerator
            else:
                QtWidgets.QMessageBox.warning(self.mainwindow, "Warning", "Unknown segmentation model")
                return
        else:
            QtWidgets.QMessageBox.warning(self.mainwindow, "Warning", "No segmentation model")
            return

        self.auto_segment_predictor = SamAutomaticMaskGenerator(
            model=self.mainwindow.segany.predictor.model,
            points_per_side=self.SpinBox_points_per_side.value(),
            pred_iou_thresh=self.DoubleSpinBox_pred_iou_thresh.value(),
            stability_score_thresh=self.DoubleSpinBox_stability_thresh.value(),
            min_mask_region_area=self.SpinBox_min_mask_area.value(),
        )

    def predict(self):
        self.init_auto_segment_predictor()

        if self.auto_segment_predictor is None:
            return

        if self.mainwindow.current_index is None:
            return

        self.mainwindow.setEnabled(False)
        self.progressBar.setValue(0)
        self.mainwindow.repaint()

        image_path = os.path.join(self.mainwindow.image_root, self.mainwindow.files_list[self.mainwindow.current_index])
        image = np.array(Image.open(image_path).convert("RGB"))
        results = self.auto_segment_predictor.generate(image)

        self.progressBar.setMaximum(len(results))

        categories_list = list(self.mainwindow.category_color_dict.keys())

        for index, result in enumerate(results):
            segmentation = result["segmentation"]
            area = result["area"]
            bbox = result["bbox"]
            predicted_iou = result["predicted_iou"]
            point_coords = result["point_coords"]
            stability_score = result["stability_score"]
            crop_box = result["crop_box"]

            contours, hierarchy = self.mainwindow.mask_to_polygon(mask=segmentation)

            for contour in contours:
                if len(contour) < 3:
                    continue
                if self.mainwindow.scene.current_graph is None:
                    self.mainwindow.scene.current_graph = Polygon()
                    self.mainwindow.scene.addItem(self.mainwindow.scene.current_graph)

                self.mainwindow.scene.current_graph.hover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_hover"] * 255
                )
                self.mainwindow.scene.current_graph.nohover_alpha = int(
                    self.mainwindow.cfg["software"]["polygon_alpha_no_hover"] * 255
                )

                for point in contour:
                    x, y = point[0]
                    x = max(0.1, x)
                    y = max(0.1, y)
                    self.mainwindow.scene.current_graph.addPoint(QtCore.QPointF(x, y))

                if self.CheckBox_random_category.isChecked():
                    category = random.choice(categories_list)
                else:
                    category = "UNKNOW"
                group = self.mainwindow.current_group

                self.mainwindow.scene.current_graph.set_drawed(
                    category,
                    group,
                    False,
                    "",
                    QtGui.QColor(
                        self.mainwindow.category_color_dict.get(category, "#6F737A")
                    ),
                    len(self.mainwindow.polygons) + 1,
                )

                # 添加新polygon
                self.mainwindow.polygons.append(self.mainwindow.scene.current_graph)
                self.mainwindow.annos_dock_widget.listwidget_add_polygon(
                    self.mainwindow.scene.current_graph
                )
                self.mainwindow.scene.current_graph = None
            if self.mainwindow.group_select_mode == "auto":
                self.mainwindow.current_group += 1
                self.mainwindow.categories_dock_widget.lineEdit_currentGroup.setText(
                    str(self.mainwindow.current_group)
                )

            self.progressBar.setValue(index + 1)

        self.mainwindow.setEnabled(True)
