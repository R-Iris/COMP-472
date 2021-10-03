import matplotlib.pyplot as plt
import pandas as pd
import os
classifications = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']
instances = [len(os.listdir("BBC/business")),
                len(os.listdir("BBC/entertainment")),
                len(os.listdir("BBC/politics")),
                len(os.listdir("BBC/sport")),
                len(os.listdir("BBC/tech"))]
plt.bar(classifications, instances)
plt.xlabel('Classification')
plt.ylabel('Instances')
plt.title('BBC Distribution')
plt.savefig("BBC-distribution.pdf")
