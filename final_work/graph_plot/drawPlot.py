import matplotlib.pyplot as plt
import math

# DCG for all techniques
ts = [1,5,10,50,100,300,600]
abDCG = [0.88028758, 0.88411406, 0.88314027, 0.88530777, 0.88343944, 0.8844624, 0.88510801]
rfDCG = [0.88088834, 0.88467628, 0.88553513, 0.88708591, 0.88805239, 0.88634021, 0.88741406]
xgDCG = [0.88296284, 0.88381029, 0.88384027, 0.88554312, 0.8828267, 0.88201774, 0.88129638]

#ts = [math.log10(i) for i in ts]

plt.figure(1)
plt.subplot(211)
plt.plot(ts, rfDCG)
plt.plot(ts, abDCG)
plt.plot(ts, xgDCG)
#plt.xlabel('Tree Size')
plt.ylabel('DCG Mean Score')
plt.legend(['Random Forest', 'AdaBoost', 'XGBoost'], loc='upper right')

# RMSE for all techniques
abRMSE = [0.48869603, 0.48883938, 0.49062925, 0.49307776, 0.49074649, 0.48890829, 0.48849815]
rfRMSE = [0.50511016, 0.48098642, 0.4777869, 0.47515415, 0.47389116, 0.47374845, 0.47362134]
xgRMSE = [0.52240614, 0.49782962, 0.48646472, 0.48132337, 0.48282088, 0.49099913, 0.49914065]
plt.subplot(212)
plt.plot(ts, rfRMSE)
plt.plot(ts, abRMSE)
plt.plot(ts, xgRMSE)
plt.xlabel('Tree Size')
plt.ylabel('RMSE Mean Score')
plt.legend(['Random Forest', 'AdaBoost', 'XGBoost'], loc='upper right')

# Only Random forest
plt.figure(2)
plt.subplot(211)
plt.plot(ts, rfDCG)
plt.ylabel('DCG Mean Score')
plt.legend(['Random Forest', 'AdaBoost', 'XGBoost'], loc='right')

plt.subplot(212)
plt.plot(ts, rfRMSE)
plt.xlabel('Tree Size')
plt.ylabel('RMSE Mean Score')
plt.legend(['Random Forest', 'AdaBoost', 'XGBoost'], loc='right')

plt.show()
