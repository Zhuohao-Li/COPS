import matplotlib.pyplot as plt

# Data for the three trends
our = [28,35.1,36.1,38.1,38.7,39,41,43.1,43.1,43.1]
reflexion = [24,28,30.7,32.3, 35.1,37.1,38.7,38.7,38.7,38.7]
rap = [20,21.1,22.6,26.7,30,33.3,36.4,37.5,40,40]
lats = [20,30,32.3,34.5,37.2,38.6,40,40,40,40]
# our=[x/1.34 for x in our]
#reflexion=[x/1.34 for x in reflexion]
# Define the x-axis values (time points or similar)
x = list(range(0,10))

# Plotting the trends

plt.figure(figsize=(6,4))



plt.plot(x, reflexion, label='Reflexion', marker='s')
plt.plot(x, rap, label='RAP', marker='v')
plt.plot(x, lats, label='LATs', marker='^', color='green')
plt.plot(x, our, label='CoPS', marker='o',color='red')
plt.ylim(bottom=0)
# Adding titles and labels
#plt.title('Results on Llama3.1 70b')
plt.xlabel('Trials',fontsize=25)
plt.ylabel('Success rate (percentage)',fontsize=15) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)

# Displaying legend

plt.legend(loc='lower right',fontsize=20)

# Show the plot
plt.grid(False)

plt.tight_layout()

# 保存为PDF，使用bbox_inches参数去掉白边
plt.savefig("alf8b.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()


# 70b

# Data for the three trends
our70b = [44.5,55.2,57.1,59,60.5,61,61.2,61.2,61.2,61.2]
reflexion70b = [40,40.8,41.3,41.8,42.3,42.8,43,43.2,43.4,43.5]
rap70b     = [30.8,33.3,36.4,37.5,38.9,40,42.9,50,55.6,57.1]
lats70b = [30,40,42.3,45.1,47, 48.5, 49.2,50,50,50]
# our=[x/1.34 for x in our]
#reflexion=[x/1.34 for x in reflexion]
# Define the x-axis values (time points or similar)
x = list(range(0,10))

# Plotting the trends

plt.figure(figsize=(6,4))



plt.plot(x, reflexion70b, label='Reflexion', marker='s')
plt.plot(x, rap70b, label='RAP', marker='v')
plt.plot(x, lats70b, label='LATs', marker='^', color='green')
plt.plot(x, our70b, label='CoPS', marker='o',color='red')
plt.ylim(bottom=0)
# Adding titles and labels
#plt.title('Results on Llama3.1 70b')
plt.xlabel('Trials',fontsize=25)
plt.ylabel('Success rate (percentage)',fontsize=15) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)

# Displaying legend

plt.legend(loc='lower right',fontsize=20)

# Show the plot
plt.grid(False)

plt.tight_layout()

# 保存为PDF，使用bbox_inches参数去掉白边
plt.savefig("alf70b.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()