# 时间 : 2021/08/18 9:18
# 作者 : 张志高
# 文件 : weapon_test
# 项目 : 墨子联合作战智能体研发平台
# 版权 : 北京华戍防务技术有限公司

from mozi_ai_sdk.test.utils.test_framework import TestFramework


class TestWeapon(TestFramework):
    """测试武器"""

    def test_get_summary_info(self):
        """获取精简信息, 提炼信息进行决策"""
        info = self.weapon_1.get_summary_info()
        self.env.step()
        self.assertEqual(info.get('name'), self.weapon_1.strName)

