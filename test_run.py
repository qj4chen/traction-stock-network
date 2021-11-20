from utils import *
import matplotlib.pyplot as plt

start_date = '2013-04-01'
end_date = '2020-12-31'

back_test_result = back_test(start=start_date,
                             end=end_date,
                             principal=1_000_000,
                             opening_rule='等数量',
                             stock_pool=STOCK_LIST)

plt.clf()
back_test_result.plot()
plt.savefig('./收益率曲线.png')