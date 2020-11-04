#!/usr/bin/env python
# coding: utf-8

# # Methods for comparing two observed confusion matrices
# 
# This notebook implements tests for two observed confusion matrices to evaluate two assignation procedures.
# 
# The notebook implementation is based on a paper: José Rodríguez-Avi, Francisco Javier Ariza-López, Mª Virtudes Alba-Fernández - Methods for comparing two observed confusion matrices, Proceedings 2018, The 21st AGILE International Conference on Geographic Information Science [download link](https://agile-online.org/conference_paper/cds/agile_2018/posters/96%20Poster%2096.pdf)
# 
# Python implementation by Nikola Mirkov (largeddysimulation@gmail.com)

# ### Create a class for confusion matrix comparison

# In[4]:


import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

class confmatcomp:
    '''
    Purpose:
    A class implementing confusion matrix comparison.
    
    Desription:
    You need to give two matrices as input argument and run various tests
    implemented as methods of the class.
    
    Implemented test methods are:
    1) Single binomial contrast (called by: Test_instance_name.single_binomial_contrast(args) )
    2) Mutiple binomial contrast (called by: Test_instance_name.multiple_binomials_contrast(args) )
    3) Overall chi-square test (called by Test_instance_name.chi_square(args))
    
    Reference: 
    J. Rodrigues-Avi et al., "Methods for comparing two observed conusion matrices",...(201X)
    
    '''
    def __init__(self, MatA:np.array, MatB:np.array):
        self.A = MatA
        self.B = MatB
        
    def get_nm(self):
        n = np.sum(self.A[:,:])
        m = np.sum(self.B[:,:])
        print("Total number of elements classified in A: {}, and B: {}".format(n,m) )
        
    def single_binomial_contrast(self,tol):
        '''
        Purpose:
        This test compares the global proportion of 
        concordant elements in both classifications.
        Null hypothesis is that these two proportions are the same.
           
        Arguments:
        tol (input) - tolerance level for rejection of null hypothesis.
        '''
        print("Null hypothesis is that these two proportions are the same.")
        n = np.sum(self.A[:,:])
        piA = sum(np.diag(self.A))/float(n)
        print("Concordant elements proportion in 1st matrix: {:.6f}".format(piA) )
        m = np.sum(self.B[:,:])
        piB = sum(np.diag(self.B))/float(m)
        print("Concordant elements proportion in 2nd matrix: {:.6f}".format(piB) )
        if ( piA-piB < tol ):
            print("Global proportion of concordant elements is the same!")
        else:
            print("Global proportion of concordant elements differs by: {}, \nNull hypothesis rejected!". 
                  format( abs(piA-piB) ) ) 
            
    def multiple_binomials_contrast(self, tol, column=True):
        '''
        Function: multiple_binomials_contrast
        
        Purpose:
        This test makes individual tests by rows or columns.
        Default is test by columns.
           
        Description:
        We split problem into k-null hypotheses.
        Then we find Z estimator and fit it to normal distribution.
        After that k p-values are obtained, where k is the matrix size.
        p-values are then corrected using Bonferonni's correction to assure
        Type I error level.
           
        Arguments:
        tol - tolerance for pairwise comparison (eg. tol=1e-6)
        column - (optional) True/False, if true column-wise estimators
        are produced, if false rwo-wise estimators are produced.
        '''
        k = self.A.shape[1]
        Z = np.zeros(k)
        if(column):
            print("\nColumn-wise estimators for 1st and 2nd matrix and k-th null hypothesis testing:\n")
            for j in range( k ):
                n_plj = np.sum( self.A[:,j] )
                piAj = self.A[j,j] / float(n_plj)
                m_plj = np.sum( self.B[:,j] )
                piBj = self.B[j,j] / float(m_plj)            
                print( "Column {} : {} {} {}". format(j, piAj, piBj, (abs(piAj-piBj)<tol) ) )
                Z[j] = (piAj-piBj)/np.sqrt( piAj*(1-piAj)/ float(n_plj) + piBj*(1-piBj)/ float(m_plj) )
        else:
            print("\nRow-wise estimators for 1st and 2nd matrix and k-th null hypothesis testing:\n")
            for j in range( k ):
                n_plj = np.sum( self.A[j,:] )
                piAj = self.A[j,j] / float(n_plj)
                m_plj = np.sum( self.B[j,:] )
                piBj = self.B[j,j] / float(m_plj)            
                print( "Row {} : {} {} {}". format(j, piAj, piBj, (abs(piAj-piBj)<tol) ) )
                Z[j] = (piAj-piBj)/np.sqrt( piAj*(1-piAj)/ float(n_plj) + piBj*(1-piBj)/ float(m_plj) )
        #print(Z)
        mu, std = sps.norm.fit(Z)
        #print(mu,std)
        # Plot distribution
        plt.hist(Z, bins = 5, density=True, alpha=0.8, color='b')
        # Plot PDF of normal distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin,xmax, 100)
        p = sps.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=1)
        title = "Fit results for Z-vector: mu=%.3f, std=%.3f" % (mu,std)
        plt.title(title)
        plt.show()
        
        # Cumulative distribution for the test statistics
        cdfts = sps.norm.cdf( abs(Z) )
        pval = 2*(1-cdfts) # p-value for two-sided test
        pvalcorr = pval/k  # Bonferonni's correction
        print("\nk p-values and Bonferonni's correction:\n")
        for j in range(k):
            print("{} : {:.4f} {:.4f}".format(j, pval[j], pvalcorr[j]))
            
    def chi_square(self,column=True):
        '''
        Purpose:
        Peform overall chi-square test.
        
        Desription:
        Performs a chis-aquare test on sqare of Z vector obtained in 
        the way described in multiple_binomials_contrast test.
        This vector follows Chi-squared distribution.
        If observed statistics doesn't obey this - the Null hypothesis
        of whole equality of given confusion matrices is rejected.
        Further inquiry may give rejection causes.
        
        Arguments:
        column - (optional) True/False, if true column-wise estimators
        are produced, if false rwo-wise estimators are produced.
        '''
        k = self.A.shape[1]
        Z = np.zeros(k)
        if(column):
            #print("\nColumn-wise estimators for 1st and 2nd matrix and k-th null hypothesis testing:\n")
            for j in range( k ):
                n_plj = np.sum( self.A[:,j] )
                piAj = self.A[j,j] / float(n_plj)
                m_plj = np.sum( self.B[:,j] )
                piBj = self.B[j,j] / float(m_plj)            
                #print( "Column {} : {} {} {}". format(j, piAj, piBj, (abs(piAj-piBj)<tol) ) )
                Z[j] = (piAj-piBj)/np.sqrt( piAj*(1-piAj)/ float(n_plj) + piBj*(1-piBj)/ float(m_plj) )
        else:
            #print("\nRow-wise estimators for 1st and 2nd matrix and k-th null hypothesis testing:\n")
            for j in range( k ):
                n_plj = np.sum( self.A[j,:] )
                piAj = self.A[j,j] / float(n_plj)
                m_plj = np.sum( self.B[j,:] )
                piBj = self.B[j,j] / float(m_plj)            
                #print( "Row {} : {} {} {}". format(j, piAj, piBj, (abs(piAj-piBj)<tol) ) )
                Z[j] = (piAj-piBj)/np.sqrt( piAj*(1-piAj)/ float(n_plj) + piBj*(1-piBj)/ float(m_plj) )
        
        #Z-squared:
        Zsq = Z*Z
        chisq, pval = sps.chisquare(Zsq)
        print("Chi-squred test statistic: {:.3f}, p-value of the test: ".format(chisq), pval)
    
    def multinomial_distance_bootstrap(self):
        '''
        Purpose:
        Peform multinomial distance bootstrap test.
        
        Desription:
        Obtain vectors from confusion matrices and estimate distance 
        between these vectors.
        
        Arguments:

        '''
        a = (self.A).flatten()
        b = (self.B).flatten()
        dist = np.sum( (np.sqrt(a)-np.sqrt(b))**2 )
        n = np.sum(self.A[:,:])
        m = np.sum(self.B[:,:])
        Tnm = 4*(n+m)*dist
        print( "Estimate distance between vectors: {:.6f}, {:.6f}".format(dist,Tnm) )        

