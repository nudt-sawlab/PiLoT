from pixloc.utils.transform import euler_angles_to_matrix_ECEF
euler_angles = [43.71077368954762, 0.3998095795211294, -100.28029029820274]
translation = [7.623107921826066, 46.74151506260668, 1144.704415665939]
T_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
print(T_in_ECEF_c2w)
# 7.623116435370997, 46.74151416097179, 1144.9536383142695