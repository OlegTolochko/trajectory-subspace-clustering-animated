from manim import *
from manim import config
import numpy as np
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
from manim_ml.neural_network.animations.dropout import make_neural_network_dropout_animation
from manim_ml import ManimMLConfig

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
        
        im = ImageMobject("images/final_frame.png").scale(0.6)
        
        self.add(im)
        
        self.wait(3)
        
        self.play(
            self.camera.frame.animate.shift(RIGHT * 2)
            , run_time=1.5
        )
        
        self.wait(3)
        
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

        matrices_group = VGroup(*matrix_mobjects).arrange(DOWN, buff=0.15).next_to(im, RIGHT, buff=1)
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
        
        
        self.play(
            self.camera.frame.animate.shift(RIGHT * 9.5)
            , run_time=1.5
        )

        encoder = Polygon(
            [-3.5, -1.5, 0], [-2, -0.5, 0], [-2, 0.5, 0], [-3.5, 1.5, 0],
            color=encoder_color, fill_opacity=0.2, stroke_color=encoder_color, stroke_width=2
        ).next_to(matrices_group, RIGHT, buff=1)
        encoder_text = Paragraph("Feature", "Extractor").scale(0.5).move_to(encoder.get_center())
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

        encoder_group = VGroup(encoder, encoder_text)

        self.play(
            self.camera.frame.animate.shift(RIGHT * 6.25)
            , run_time=1.5
        )
        self.wait(0.5)
        
        reverse_feature_coloring = []
        for feature_vec in feature_vector_mobjects:
            reverse_feature_coloring.append(FadeToColor(feature_vec, GRAY))
        
        self.play(FadeOut(*created_mobjects), reverse_feature_coloring)

        self.wait(2)

        
        # -- subspace estimator box + modules
        subspace_estimator_color = BLUE_D
        basis_functions_color = ORANGE
        final_basis_B_color = TEAL_D

        estimator_box_width = 6.0
        estimator_box_height = 5.0 
        subspace_estimator_module = RoundedRectangle(
            width=estimator_box_width, height=estimator_box_height, corner_radius=0.2,
            color=subspace_estimator_color, fill_opacity=0.1, stroke_width=2.5
        )
        subspace_estimator_module.next_to(encoder_group, RIGHT, buff=5.75)

        subspace_estimator_title_text = Text("Subspace Estimator (g)", font_size=20, weight=BOLD).next_to(subspace_estimator_module, UP, buff=0.25)

        mlp_nn = NeuralNetwork([
            FeedForwardLayer(3, neuron_radius=0.12).scale(0.8),
            FeedForwardLayer(5, neuron_radius=0.12).scale(0.8),
            FeedForwardLayer(4, neuron_radius=0.12).scale(0.8)
        ], layer_spacing=0.35
        )
        mlp_nn.scale(0.75)
        mlp_nn.move_to(subspace_estimator_module.get_center() + UP * (estimator_box_height / 4.5) + LEFT*1 )
        
        mlp_label_text = Text("MLP ω(f)", font_size=16).next_to(mlp_nn, UP, buff=0.15)

        basis_func_box_visual = RoundedRectangle(
            width=3, height=estimator_box_height * 0.30, corner_radius=0.2,
            color=basis_functions_color, fill_opacity=0.2, stroke_width=2
        )
        basis_func_box_visual.move_to(subspace_estimator_module.get_center() + DOWN * (estimator_box_height / 5.5) + LEFT*1)
        basis_func_text_label = Text("Basis Functions h(t)", font_size=16).move_to(basis_func_box_visual.get_center())
        
        
        times_box = RoundedRectangle(
            width=1, height=1, corner_radius=0.2,
            color=GREEN, fill_opacity=0.1, stroke_width=2.5
        ).next_to(subspace_estimator_module.get_left() + LEFT*1.75 + DOWN * (estimator_box_height / 5.5))
        
        times_text = Paragraph("Num", "Frames", alignment='center', font_size=16).move_to(times_box.get_center())
    
        
        combination_symbol = MathTex(r"\times", font_size=50, color=BLACK)
        combination_symbol.move_to(subspace_estimator_module.get_center() + RIGHT*2)
        combination_circle = Circle(radius=0.185, color=BLACK).move_to(combination_symbol.get_center())


        basis_B_matrix_content = [ ["B_1^{2 \\times r}"], ["\\vdots"], ["B_F^{2 \\times r}"] ]
        subspace_basis_B_matrix = Matrix(basis_B_matrix_content, h_buff=1.0, v_buff=0.7).scale(0.45)
        subspace_basis_B_matrix.next_to(subspace_estimator_module, RIGHT, buff=0.6).set_color(final_basis_B_color)
        basis_B_title_text = Text("Subspace Basis B", font_size=20, weight=BOLD).next_to(subspace_basis_B_matrix, UP, buff=0.25)

        self.play(
            FadeIn(subspace_estimator_module, shift=RIGHT*0.2),
            Write(subspace_estimator_title_text),
            run_time=1.0
        )
        self.play(
            Write(mlp_label_text),
            Create(mlp_nn),
            Write(basis_func_text_label),
            FadeIn(basis_func_box_visual),
            Write(combination_symbol),
            Create(combination_circle),
            run_time=1.5
        )
        self.wait(1)
        
        # -----

        source_feature_vector_obj = feature_vector_mobjects[1] 
        animated_f_input_copy = source_feature_vector_obj.copy().set_color(source_feature_vector_obj.get_color())
        
        mlp_input_target_point = mlp_nn.get_left() + LEFT * 0.3
        
        arrow_f_to_mlp_input = Arrow(
            source_feature_vector_obj.get_right(), 
            mlp_input_target_point, 
            buff=0.1, stroke_width=3, max_tip_length_to_length_ratio=0.15, 
            color=source_feature_vector_obj.get_color()
        )
        
        
        self.play(GrowArrow(arrow_f_to_mlp_input))
        self.play(
            animated_f_input_copy.animate.move_to(mlp_input_target_point).scale(0.6), 
            run_time=0.75
        )
        
        self.play(FadeOut(arrow_f_to_mlp_input), FadeOut(animated_f_input_copy), run_time=0.3)
        
        forward_pass_anim = mlp_nn.make_forward_pass_animation(
            run_time=1.5,
            passing_flash_color=YELLOW
        )
        self.play(forward_pass_anim)
        
        
        
        # -- basis
        self.play(FadeIn(times_box), Write(times_text))
        arrow_num_frames_to_basis = Arrow(
            times_box.get_right(), 
            basis_func_box_visual.get_left(), 
            buff=0.1, stroke_width=2, max_tip_length_to_length_ratio=0.15, 
            color=BLACK
        )
        
        self.play(GrowArrow(arrow_num_frames_to_basis))
        
        mlp_output_point = mlp_nn.get_right() + RIGHT * 0.2
        
        
        # -- subspace basis
        self.play(
            Write(basis_B_title_text), 
            FadeIn(subspace_basis_B_matrix, shift=RIGHT*0.1), 
            run_time=1.0
        )
        self.wait(2)
    

        final_pipeline_view = VGroup(
            encoder_group, 
            feature_vector_mobjects[0],
            subspace_estimator_module, 
            subspace_basis_B_matrix,
            basis_B_title_text,
            subspace_estimator_title_text
        )
        final_pipeline_view = VGroup(*[m for m in final_pipeline_view if m is not None and m in self.mobjects])

        self.wait(5)
