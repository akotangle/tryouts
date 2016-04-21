
'''PREDICTIVE ANALYSIS '''
'''
1. Facility provided by all stores are same.
2. The product sold are of same brand.
3. Product sold are at same rate.
4. Discount and offers provided are same throughout.
5. Tax collected is same in all stores.
'''


import scipy, numpy
import scipy.optimize, scipy.stats
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
import statsmodels.stats
import statsmodels.stats.stattools as stools
import statsmodels.api as sm
import statsmodels.formula.api as smf

#model of grocery store
'''To predict sales of a grocery store'''
'''
the grocery store sell 4 items A,B,C,D
A and B has higher probabiliity of selling at the start of month
C has higher probabilty of selling on weekends
D is independent of the day of month'''
Pa1=0.8 #probability of a when sales are higher 
Pa2=0.3#probability of a when sales are lower
Pb1=0.7#probability of b when sales are higher
Pb2=0.4#probability of b when sales are lower
Pc1=0.9#probability of c when sales are higher
Pc2=0.5#probability of c when sales are lower
'''' sales dependence on advertising'''
P_ad1=0.8#probability if the advertisement is within 1 km radius of store
P_ad2=0.4#probability if the advertisement is beyond 1 km radius of store

'''sales depenence on Amount to be paid'''
p_c=0.9#Probability if Amount to be paid is less than 500
p_c=0.8#Proability if Amount to be paid is greater than 500
'''
if amount to be paid is greater than 500 then transaction is to be done by Card
if amount to be paid is less than 500 then transaction is to be done by Cash
'''
def fitdata(f, Xdata,Ydata,Errdata, pguess, ax=False, ax2=False):
    '''
    popt = vector of length N of the optimized parameters
    pcov = Covariance matrix of the fit
    perr = vector of length N of the std-dev of the optimized parameters
    p95 = half width of the 95% confidence interval for each parameter 
    p_p = vector of length N of the p-value for the parameters being zero
    (if p<0.05, null hypothesis rejected and parameter is non-zero) 
    chisquared = chisquared value for the fit
    chisquared_red = chisquared/degfreedom
    chisquare = (p, chisquared, chisquared_red, degfreedom) 
    p = Probability of finding a chisquared value at least as extreme as the one shown
    chisquared_red = chisquared/degfreedom. value should be approx. 1 for a good fit. 
    R2 = correlation coefficient or proportion of explained variance 
    R2_adj = adjusted R2 taking into account number of predictors 
    resanal = (p, w, mean, stddev)
    Analysis of residuals 
    p = Probability of finding a w at least as extreme as the one observed (should be high for good fit) 
    w = Shapiro-Wilk test criterion 
    mean = mean of residuals 
    p_res = probability that the mean value obtained is different from zero merely by chance 
    F = F-statistic for the fit msm/msE. 
    Null hypothesis is that there is NO Difference between the two variances. 
    p_F = probability that this value of F can arise by chance alone.
    p_F < 0.05 to reject null hypothesis and prove that the fit is good.
    dw = Durbin_Watson statistic (value between 0 and 4). 
    2 = no-autocorrelation. 0 = .ve autocorrelation, 4 = -ve autocorrelation. 
'''    
    
    def error(p,Xdata,Ydata,Errdata):
        Y=f(Xdata,p)
        residuals=(Y-Ydata)/Errdata
        return residuals
    res=scipy.optimize.leastsq(error,pguess,args=(Xdata,Ydata,Errdata),full_output=1)
    (popt,pcov,infodict,errmsg,ier)=res
    perr=scipy.sqrt(scipy.diag(pcov))

    M=len(Ydata)
    N=len(popt)
    #Residuals
    Y=f(Xdata,popt)
    residuals=(Y-Ydata)/Errdata
    meanY=scipy.mean(Ydata)
    squares=(Y-meanY)/Errdata
    squaresT=(Ydata-meanY)/Errdata
    
    SSM=sum(squares**2) #Corrected Sum of Squares
    SSE=sum(residuals**2) #Sum of Squares of Errors
    SST=sum(squaresT**2)#Total Corrected sum of Squares
    
    DFM=N-1 #Degree of Freedom for model
    DFE=M-N #Degree of Freedom for error
    DFT=M-1 #Degree of freedom total
    
    MSM=SSM/DFM #Mean Squares for model(explained Variance)
    MSE=SSE/DFE #Mean Squares for Error(should be small wrt MSM) unexplained Variance
    MST=SST/DFT #Mean squares for total
    
    R2=SSM/SST #proportion of unexplained variance 
    R2_adj= 1-(1-R2)*(M-1)/(M-N-1) #Adjusted R2
    
    #t-test to see if parameters are different from zero
    t_stat=popt/perr #t-stat for popt different from zero
    t_stat=t_stat.real
    p_p= 1.0-scipy.stats.t.cdf(t_stat,DFE) #should be low for good fit
    z=scipy.stats.t(M-N).ppf(0.95)
    p95=perr*z
    #Chi-Squared Analysis on Residuals
    chisquared=sum(residuals**2)
    degfreedom=M-N
    chisquared_red=chisquared/degfreedom
    p_chi2=1.0-scipy.stats.chi2.cdf(chisquared, degfreedom)
    stderr_reg=scipy.sqrt(chisquared_red)
    chisquare=(p_chi2,chisquared,chisquared_red,degfreedom,R2,R2_adj)
    
    #Analysis of Residuals
    w, p_shapiro=scipy.stats.shapiro(residuals)
    mean_res=scipy.mean(residuals)
    stddev_res=scipy.sqrt(scipy.var(residuals))
    t_res=mean_res/stddev_res #t-statistics
    p_res=1.0-scipy.stats.t.cdf(t_res,M-1)
    
    F=MSM/MSE
    p_F=1.0-scipy.stats.f.cdf(F,DFM,DFE)
    
    dw=stools.durbin_watson(residuals)
    resanal=(p_shapiro,w,mean_res,p_res,F,p_F,dw)
    
    if ax:
        formataxis(ax)
        ax.plot(Ydata,Y,'ro')
        ax.errorbar(Ydata,Y,yerr=Errdata, fmt='.')
        Ymin,Ymax=min((min(Y),min(Ydata))),max((max(Y),max(Ydata)))
        ax.plot([Ymin,Ymax],[Ymin,Ymax],'b')
        
        ax.xaxis.label.set_text('Data')
        ax.yaxis.label.set_text('Fitted')
        sigmay,avg_stddev_data=get_stderr_fit(f,Xdata,popt,pcov)
        Yplus=Y+sigmay
        Yminus=Y-sigmay
        ax.plot(Y,Yplus,'c',alpha=0.6,linestyle='--',linewidth=0.5)
        ax.plot(Y,Yminus,'c',alpha=0.6,linestyle='==',linewidth=0.5)
        ax.fill_between(Y,Yminus,Yplus,facecolor='cyan',alpha=0.5)
        titletext='Parity plot for fit.\n'
        titletext+=r'$r^2$=%5.2f,$r^2_{adj}$=%5.2f,$p_{shapiro}$=%5.2f,$Durbin-Watson=%2.1f'
        titletext+='\n F=%5.2f,$p_F$=%3.2e'
        titletext+='$\sigma_{err}^{reg}$=%5.2f'
        
        #ax.title.set_text(titletext%(R2, R2_adj, avg_stddev_data, chisquared_red, p_chi2, stderr_reg))
        ax.figure.canvas.draw()
    
    if ax2:
        formataxis(ax2)
        ax2.plot(Y,residuals,'ro')
        ax2.xaxis.label.set_text('Fitted Data')
        ax2.yaxis.label.set_text('Residuals')
        
        titletext='Analysis of Residuals\n'
        titletext+=r'mean=%5.2f,$p_{res}$=%5.2f,$p_{shapiro}$=%5.2f,$Durbin-Watson$=%2.1f'
        titletext+='\n F=%5.2f,$p_F$=%3.2e'
        ax2.title.set_text(titletext%(mean_res,p_res,p_shapiro,dw,F,p_F))
    return popt,pcov,perr, p95, p_p,chisquare, resanal,dw
    
def get_stderr_fit(f,Xdata,popt, pcov):
    Y=f(Xdata,popt)
    listdY=[]
    for i in xrange(len(popt)):
        p=popt[i]
        dp=abs(p)/1e6 + 1e-20
        popt[i] += dp
        Yi = f(Xdata,popt)
        dY = (Yi-Y)/dp
        listdY.append(dY)
        popt[i] -= dp
    listdY=scipy.array(listdY)
    left=scipy.dot(listdY.T,pcov)
    right=scipy.dot(left,listdY)
    sigma2y=right.diagonal()
    mean_sigma2y=scipy.mean(right.diagonal())
    M=Xdata.shape[1]
    N=len(popt)
    avg_stddev_data=scipy.sqrt(M*mean_sigma2y/N)
    sigmay=scipy.sqrt(sigma2y)
    return sigmay,avg_stddev_data

def formataxis(ax):
    ax.xaxis.label.set_fontname('Georgia')
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontname('Georgia')
    ax.yaxis.label.set_fontsize(12)
    ax.title.set_fontname('Georgia')
    ax.title.set_fontsize(12)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

#IMPORTING DATA FROM EXCEL

def import_data(xlfile,sheetname):
    df=pd.read_excel(xlfile,sheetname=sheetname)
    return df

#PLOTTING GRAPH

def prepare_data(df,Criterion,Predictors,Error=False):
    Y=scipy.array(df[Criterion])
    if Error:
        Errdata=scipy.array(df[Error])
    else: 
        Errdata=scipy.ones(len(Y))
    Xdata=[]
    for X in Predictors:
        X=list(df[X])
        Xdata.append(X)
    Xdata=scipy.array(Xdata)
    return Xdata, Y, Errdata

if __name__=='__main__':
    fig=plt.figure()
    ax=fig.add_subplot(111)
    fig.show
    
    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    fig2.show()
    
#PREDICTIVE ANALYSIS USING MULTIPLE REGRESSION    
    
    def f(X,p):
        (Q_A,Q_B,Q_C,Q_D,Time,Advertising)=X
        # probability depends on this 
    
        if (day<=10):
            Pa=0.8 #probability of selling of A when day is greater than 10
            Pb=0.7 #probability of selling of B when day is greater than 10
               
        elif (day>10 and day<=31):
            Pa=0.3 #probability of selling of A when day is less than 10
            Pb=0.4 #probability of selling of B when day is less than 10
        
        
       
       
        if (answer == "Y"): 
            Pc=0.9  #probability of selling of C when day is weekend
            
        elif (answer == "N"):
            
            Pc=0.5  #probability of selling of C when day is weekday
            
       
       
        if Advertise<=3:
            p_ad = 0.8 # Probability that advertisement will effect sales if it is within 3Km from store
            
        elif Advertise>3:
            p_ad = 0.4  # Probability that advertisement will effect sales if it is beyond 3Km from store
    
        
        
        if amount <=500:
            P_t = 0.9 #Probability of happening of transaction if total bill is less than 500
            
        elif amount >500:
            P_t = 0.8  #Probability of happening of transaction if total bill is greater than 500
        
        
        Y=((p[1]*Pb*Q_B)-(p[0]*Pa*Q_A)+(p[2]*Pc*Q_C)+(p[3]*Q_D*0.9)+(p[4]*Time)+(p[5]*Advertising*p_ad))*P_t
                
        return Y
        
    
    df=import_data('project.xlsx','Sheet2')
    Xdata,Ydata, Errdata=prepare_data(df,'Sales',('Q_A','Q_B','Q_C','Q_D','Time','Advertising',),Error='err')
    #print Xdata
    #print Ydata
    #print Errdata
    
    #initial Guess
    N=6
    pguess=N*[0.0]
    day=input("Enter day of month :- ")
    answer = str(raw_input("enter 'Y' for weekend and  for 'N' for weekday :- " ))
    Advertise = int(input("Enter distance of advertisement from store in Km :- "))
    amount = int(input ("Enter total amount to be transacted :- " ))
    popt,pcov,perr,p95,p_p,chisquare,dw,resanal=fitdata(f,Xdata,Ydata,Errdata,pguess,ax=ax,ax2=ax2)
    
   
   
#PRINTING RESULTS    
    
    print "                                                  "
    print "------------------RESULTS-------------------------------------------"
    print "                                                  "
    print "A. Multiple Regression__________________________________________                           "
    print "                                                  "
    print "1. Partial effect of Quantity of Product A sold on sales =" ,round(popt[0],2)
    print "2. Partial effect of Quantity of Product B sold on sales =" ,round(popt[1],2)
    print "3. Partial effect of Quantity of Product C sold on sales =" ,round(popt[2],2)
    print "4. Partial effect of Quantity of Product D sold on sales =" ,round(popt[3],2)
    print "5. Partial effect of time for which store is open on sales =" ,round(popt[4],2)
    print "6. Partial effect of Advertisement on sales is = " ,round(popt[5],2)   
    print " Advrtisement of store is must as the partial effect of Adverising is greatest" 
    print "                                                     "
    print " B. Error Analysis of fitted data.______________________________"
    print "                  "
    print " 1. Pshapiro for the fitted data = " ,round(dw[0],2)
    print " 2. Durbin-Watson constant  = " , round(dw[6],2)
    print " 3. mean of residuals = " ,round(dw[2],2)
    print " 4. F-statistic for the fit msm/msE = " , round(dw[4] ,2)
    print " 5. P_res = " ,round(dw[3],2)
    print " 6. P_f = " ,dw[5]                                                         
    print "-----------------THANK YOU--------------------------------------------"

    
    