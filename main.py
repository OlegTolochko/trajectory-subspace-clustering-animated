from manim import *
from manim import config
import numpy as np

# --- Configuration ---
config.background_color = WHITE
Tex.set_default(color=BLACK)
MathTex.set_default(color=BLACK)
Text.set_default(color=BLACK)
Paragraph.set_default(color=BLACK)

encoder_color = GREEN
real_colors = [RED, RED, GREEN, GREEN, BLUE, BLUE]
gray_colors = [GRAY] * 6


class TrajectorySubspaceClustering(MovingCameraScene):
    def setup(self):
        MovingCameraScene.setup(self)

    def construct(self):
        trajectory_matrix_1 = np.array([
            [0.3, 0.5, 0.8, 1.2, 1.7, 2.4, 3.1, 3.8, 4.5, 5.0],
            [1.0, 1.3, 1.5, 1.6, 1.5, 1.3, 1.0, 0.6, 0.2, 0.0]
        ])
        feature_matrix_1 = np.array([-0.4, 2.1, 3.7, 0.1, 0.7, -3.1])
        trajectory_matrix_2 = np.array([
            [0.2, 0.4, 0.7, 1.1, 1.6, 2.3, 3.0, 3.7, 4.4, 4.9],
            [0.8, 1.1, 1.4, 1.5, 1.4, 1.2, 0.9, 0.5, 0.1, -0.1]
        ])
        feature_matrix_2 = np.array([-0.3, 2.0, 3.7, 0.1, 0.7, -2.9])
        trajectory_matrix_3 = np.array([
            [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8],
            [0.5, 1.0, 1.4, 1.7, 1.9, 2.0, 1.9, 1.7, 1.4, 1.0]
        ])
        feature_matrix_3 = np.array([4.4, -0.1, 0.7, 2.1, 3.7, 0.1])
        trajectory_matrix_4 = np.array([
            [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9],
            [1.9, 1.7, 1.4, 1.0, 0.5, 0.2, 0.3, 0.6, 0.9, 1.3]
        ])
        feature_matrix_4 = np.array([4.2, -0.1, 0.6, 2.0, 3.4, 0.1])
        trajectory_matrix_5 = np.array([
            [4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2],
            [3.0, 2.8, 2.5, 2.1, 1.7, 1.3, 1.0, 0.7, 0.4, 0.1]
        ])
        feature_matrix_5 = np.array([-0.2, 3.4, 3.6, 2.3, 1.4, 2.1])
        trajectory_matrix_6 = np.array([
            [4.2, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4],
            [3.2, 3.0, 2.6, 2.2, 1.8, 1.4, 1.1, 0.8, 0.5, 0.2]
        ])
        feature_matrix_6 = np.array([-0.2, 3.3, 3.5, 2.4, 1.3, 2.0])

        all_matrices = [trajectory_matrix_1, trajectory_matrix_2, trajectory_matrix_3, trajectory_matrix_4, trajectory_matrix_5, trajectory_matrix_6]
        feature_vectors = [feature_matrix_1, feature_matrix_2, feature_matrix_3, feature_matrix_4, feature_matrix_5, feature_matrix_6]
        labels = ["Dog 1", "Dog 1", "Dog 2", "Dog 2", "Person", "Person"]

        matrix_mobjects = []
        for i, matrix in enumerate(all_matrices):
            formatted_entries = []
            for row_idx, row in enumerate(matrix):
                row_entries = []
                for j in range(3):
                    row_entries.append(f"{row[j]:.1f}")
                row_entries.append("\\ldots")
                row_entries.append(f"{row[-1]:.1f}")
                formatted_entries.append(row_entries)

            manim_matrix = Matrix(
                formatted_entries,
                left_bracket="[",
                right_bracket="]"
            ).scale(0.4)

            group = VGroup(manim_matrix)
            group.set_color(gray_colors[i])
            matrix_mobjects.append(group)

        matrices_group = VGroup(*matrix_mobjects).arrange(DOWN, buff=0.15).shift(LEFT*4)
        title = Text("Trajectory Matrices (2xF)", font_size=14).next_to(matrices_group, UP, buff=0.25)
        explanation = Text(
            "Each row represents [x, y] across F frames",
            font_size=14
        ).next_to(matrices_group, DOWN, buff=0.25)

        self.play(Write(title))
        for matrix_mob in matrices_group:
             self.play(FadeIn(matrix_mob), run_time=0.3)

        self.play(Write(explanation))
        self.wait(1)

        labels_mobjects = []
        for i, matrix_mob in enumerate(matrices_group):
            label = Text(labels[i], font_size=14).next_to(matrix_mob, RIGHT, buff=0.25).set_color(real_colors[i])
            self.play(FadeToColor(matrix_mob, color=real_colors[i]), Write(label), run_time=0.5)
            labels_mobjects.append(label)

        self.wait(2)

        fade_out_anims = []
        for i, (matrix_mob, label) in enumerate(zip(matrices_group, labels_mobjects)):
            fade_out_anims.append(FadeToColor(matrix_mob, color=gray_colors[i]))
            fade_out_anims.append(FadeOut(label))
        self.play(*fade_out_anims, run_time=0.5)

        self.wait(1)

        encoder = Polygon(
            [-3.5, -1.5, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.5, 0],
            color=encoder_color, fill_opacity=0.2, stroke_color=encoder_color, stroke_width=2
        ).next_to(matrices_group, RIGHT, buff=1)
        encoder_text = Text("Encoder").scale(0.5).move_to(encoder.get_center())
        self.play(FadeIn(encoder), Write(encoder_text))
        self.wait(1)

        feature_vector_mobjects = []
        for i, vector in enumerate(feature_vectors):
            formatted_entries = []
            for j in range(min(3, len(vector))):
                formatted_entries.append([f"{vector[j]:.1f}"])

            if len(vector) > 4:
                 formatted_entries.append(["\\vdots"])
                 formatted_entries.append([f"{vector[-1]:.1f}"])
            elif len(vector) == 4:
                 formatted_entries.append([f"{vector[3]:.1f}"])

            manim_vector = Matrix(
                formatted_entries,
                left_bracket="[",
                right_bracket="]"
            ).scale(0.4)

            group = VGroup(manim_vector)
            group.set_color(gray_colors[i])
            group.move_to(encoder.get_center()).scale(0.25).set_opacity(0)
            feature_vector_mobjects.append(group)

        self.add(*feature_vector_mobjects)

        temp_vectors_for_layout = []
        for vec_mob in feature_vector_mobjects:
             temp_copy = vec_mob.copy()
             temp_copy.scale(4)
             temp_vectors_for_layout.append(temp_copy)

        vector_rows_layout = []
        for i in range(0, len(temp_vectors_for_layout), 2):
            if i + 1 < len(temp_vectors_for_layout):
                row = VGroup(temp_vectors_for_layout[i], temp_vectors_for_layout[i + 1])
                row.arrange(RIGHT, buff=0.75)
            else:
                row = VGroup(temp_vectors_for_layout[i])
            vector_rows_layout.append(row)

        grid_layout = VGroup(*vector_rows_layout).arrange(DOWN, buff=0.75)
        grid_layout.next_to(encoder, RIGHT, buff=1)

        animations = []

        matrix_anims = [
            matrices_group[i].animate.move_to(encoder.get_center()).scale(0)
            for i in range(len(matrices_group))
        ]


        vector_anims = []
        for i, vector in enumerate(feature_vector_mobjects):
            row_idx = i // 2
            col_idx = i % 2

            target_pos = vector_rows_layout[row_idx][col_idx].get_center()

            anim = vector.animate.move_to(target_pos).set_opacity(1).scale(4)
            vector_anims.append(anim)
        
        self.play(AnimationGroup(*vector_anims, lag_ratio=0.15), AnimationGroup(*matrix_anims, lag_ratio=0.15), FadeOut(explanation), FadeOut(title),  run_time=1.5)

        grouping_animations = []
        created_mobjects = []

        for i in range(0, len(feature_vector_mobjects), 2):
            vector1 = feature_vector_mobjects[i]
            if i + 1 < len(feature_vector_mobjects):
                vector2 = feature_vector_mobjects[i+1]
            else:
                continue

            group_color = real_colors[i]
            group_label_text = labels[i]

            grouping_animations.append(FadeToColor(vector1, group_color))
            grouping_animations.append(FadeToColor(vector2, group_color))

            vector_pair_group = VGroup(vector1, vector2)

            bounding_box = SurroundingRectangle(
                vector_pair_group,
                color=group_color,
                buff=0.25,
                corner_radius=0.1
            )
            created_mobjects.append(bounding_box)
            grouping_animations.append(Create(bounding_box))

            group_label = Text(
                group_label_text,
                color=group_color,
                font_size=24 
            )
            group_label.next_to(bounding_box, RIGHT, buff=0.3)
            created_mobjects.append(group_label)
            grouping_animations.append(Write(group_label))

        self.add(*created_mobjects)

        self.play(*grouping_animations, run_time=1.5)
        self.wait(3)
