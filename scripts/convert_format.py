from pylabel import importer


dataset = importer.ImportVOC("golf_ball/Detection/Annotations",
                             path_to_images="../JPEGImages")
dataset.export.ExportToYoloV5(use_splits=True, output_path="training_1/labels", copy_images=True)
