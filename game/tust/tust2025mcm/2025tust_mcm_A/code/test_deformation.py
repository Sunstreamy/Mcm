import numpy as np
from q2 import AdvancedOctahedronClassifier


def test_invalid_deformation_type():
    """测试无效的形变类型"""
    classifier = AdvancedOctahedronClassifier()
    points = np.array([[1, 2, 3], [4, 5, 6]])

    print("测试无效的形变类型:")
    try:
        result = classifier.apply_deformation_3d(points, "invalid_type", 0.1)
        print("错误: 未捕获异常")
    except Exception as e:
        print(f"成功: 捕获异常类型: {type(e).__name__}")
        print(f"异常信息: {e}")

    print("\n测试无效的轴向参数:")
    try:
        result = classifier.apply_deformation_3d(
            points, "uniaxial_stretch", 0.1, axis="invalid_axis"
        )
        print("错误: 未捕获异常")
    except Exception as e:
        print(f"成功: 捕获异常类型: {type(e).__name__}")
        print(f"异常信息: {e}")


def test_empty_array():
    """测试空数组输入"""
    classifier = AdvancedOctahedronClassifier()
    empty_points = np.array([])
    empty_points_reshaped = empty_points.reshape(0, 3)

    print("\n测试空数组输入:")
    try:
        result = classifier.apply_deformation_3d(
            empty_points_reshaped, "uniaxial_stretch", 0.1
        )
        print(f"成功: 返回形状为 {result.shape}")
    except Exception as e:
        print(f"错误: 捕获异常类型: {type(e).__name__}")
        print(f"异常信息: {e}")


def test_normal_functionality():
    """测试正常功能"""
    classifier = AdvancedOctahedronClassifier()
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)  # 使用整数类型测试类型转换

    print("\n测试单轴拉伸 (x轴):")
    result = classifier.apply_deformation_3d(points, "uniaxial_stretch", 0.2, axis="x")
    print(f"原始点: \n{points}")
    print(f"变形后点: \n{result}")
    print(f"结果数据类型: {result.dtype}")

    print("\n测试单轴拉伸 (随机轴):")
    np.random.seed(42)  # 设置随机种子以获得可重复结果
    result = classifier.apply_deformation_3d(points, "uniaxial_stretch", 0.2)
    print(f"原始点: \n{points}")
    print(f"变形后点: \n{result}")
    print(f"结果数据类型: {result.dtype}")

    print("\n测试随机顶点位移:")
    np.random.seed(42)  # 重置随机种子
    result = classifier.apply_deformation_3d(points, "random_vertex_displacement", 0.1)
    print(f"原始点: \n{points}")
    print(f"变形后点: \n{result}")
    print(f"结果数据类型: {result.dtype}")


if __name__ == "__main__":
    test_invalid_deformation_type()
    test_empty_array()
    test_normal_functionality()
