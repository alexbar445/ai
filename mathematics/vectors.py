def vectors(q,x,y,n1,n2) :
    n1[0]=n1[0]*q*x
    n1[1]=n1[1]*q*x
    n2[0]=n2[0]*q*y
    n2[1]=n2[1]*q*y
    return n1,n2
def addvectors(n1,n2,a1) :
    return a1[0]*n1[0]+a1[1]*n2[0],a1[0]*n1[1]+a1[1]*n2[1]