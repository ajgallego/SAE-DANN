import math as m

def log2(x):
    return m.log(x)/m.log(2)

# calculate the kl divergence (measured in bits)
def kl_divergence_bits(p, q):
    cte = 0.00001
    p2 = [p[i] + cte for i in range(len(p))]
    q2 = [q[i] + cte for i in range(len(q))]

    return sum([p2[i] * log2(p2[i]/q2[i]) for i in range(len(p2))])


kl = kl_divergence_bits([0,0,0], [1,1,1])

print(kl)
