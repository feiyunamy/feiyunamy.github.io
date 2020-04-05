---
title: 动态规划、DFS和回溯
updated: 2020-04-05 12:10
---

### 一些理解
1. DP：
    - 找到最优子结构，求解过程中，将子问题的答案记录下来，求解父问题的时候可以直接得到。
    - 对应普通DP的三个步骤：
        - 定义元素数组的含义
        - 找到数组元素之间的关系式
        - 定义初始值
    - 在有些问题中，上述三个步骤对应DOWN-TOP模式，但是有些问题中的初始无法直接得出，需要在递归过程中获得，即 TOP-DOWN 的模式。通过打表方式获得。
2. DFS：
    - 简而言之就是当前问题如果无法得到结果的话，直接DFS，找到最简单的那个问题（深度最深），然后就能得到对应的结果。然后配合回溯，找到复杂问题的结果。
3. Backtracking：
    - 解决一个复杂问题时，着眼于他的子问题，通过DFS求解，但是需要保证得到子问题的解以后，复杂问题的参数可以恢复，此时，父问题便可以直接求解。
    - 恢复父问题参数的过程即称为回溯，通过求解子问题时压栈，子问题解决后弹栈的方式来实现参数的恢复。
    - 注意，恢复参数需要保证每次传递参数时，函数的变量不能被硬性改变。比如，可以是：
```python
def f(param):
    ...
    f(param + bias)
    or
    new_param = param + bias
    f(new_param)
```
但是不能这样调用，导致无法回溯参数：
```python
def f(param):
    ...
   param = param + bias
    f(param)
```
4. 一个经验是，如果初始值很容易得到，则直接使用标准 DOWN-TOP DP就可以解决，但是如果需要用到DFS+Backtracking， 则TOP-DOWN DP中的打表记录法可能会产生一定的优化效果。
    - 因为如果不打表，则需要一直DFS到能够肉眼得到结果的时候再慢慢返回，重复计算，效率过低。
### 一个例子
- 题目：[Leetcode-464 我能赢吗？](https://leetcode-cn.com/problems/can-i-win/)

  

- 代码：
  DFS + Backtracking + DP优化

  
```python
# @lc code=start
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        if maxChoosableInteger >= desiredTotal:
            return True
        all_sum =  (maxChoosableInteger + 1) * maxChoosableInteger // 2
        if all_sum < desiredTotal:
            return False
        if all_sum == desiredTotal:
            return maxChoosableInteger % 2 == 1
        nums = [i for i in range(1, maxChoosableInteger + 1)]
        # 如果 maxChoosableInteger = 5， status = 31, 二进制就是11111，可以表示5个数的被选状态
        # status = (maxChoosableInteger << 1) - 1
        self.seen = {}
        
        # self.can_win(当前剩余数字，当前目标)
        # 返回值表示这种条件限制下可以赢吗？
        return self.can_win(nums, desiredTotal)
        # 下面方式返回也可以，因为DP打表也记录了结果
        # self.can_win(nums, desiredTotal)
        # return self.seen[tuple(nums)]
    
    def can_win(self, nums, target):
        # 首先列举一些直接可以获得结果的情况，直接返回并回溯
        # 如果不行，就开始DFS
        # +++++++++++++++++++++++++++++++++++++++++++
        if nums[-1] >= target:
            return True 
        if tuple(nums) in self.seen:
            return self.seen[tuple(nums)]
        # ++++++++++++++++++++++++++++++++++++++++++++
        # DFS过程
        for i in range(len(nums)):
            # 我选完以后，如果对手输了，那我就直接赢了
            # 记录当前结果，并返回
            if not self.can_win(nums[:i] + nums[i+1:], target-nums[i]):
                # DP打表
                self.seen[tuple(nums)] = True 
                return True
        # 如果没有找到对手输掉的任何办法，那我必输
        # 记录结果，（DP打表）
        self.seen[tuple(nums)] = False 
        return False
# can = Solution().canIWin(4, 6)
# print(can)
# @lc code=end
```