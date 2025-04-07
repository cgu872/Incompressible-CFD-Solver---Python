# -*- coding:utf-8 -*-
# @data 2025/3/31

import logging,os

def MeshInfo(pointN, faceN, InnerfaceN, elementN,costTime,workdir):
    logging.basicConfig(
        filename='run.log',  # 日志文件名
        filemode='w', #'w' 表示覆盖写入
        level=logging.DEBUG,  # 日志级别
        format='%(message)s'  # 日志格式 %(asctime)s - %(levelname)s -
    )
    Casename=os.path.split(workdir)[1]
    logging.info('Running Case: '+Casename+'\n')
    logging.info('Mesh Information')
    logging.info('    points:           %d' % pointN)
    logging.info('    faces:            %d' % faceN)
    logging.info('    Inner faces:      %d' % InnerfaceN)
    logging.info('    cells:            %d' % elementN)
    logging.info("Reading mesh cost %.3f minutes\n" % costTime)

def SolveInfo(fvSchemes,timeDirectory,transportProperties,turbulenceProperties):
    logging.info('Solver Information')
    logging.info('    Time Scheme:           ' + fvSchemes['ddtSchemes'])
    logging.info('    Turbulence Model:      ' + turbulenceProperties['RAS']['RASModel'])
    logging.info('    Constant rho:          ' + str(transportProperties['rho']))
    logging.info('    Constant Dynamic Viscosity: ' + str(transportProperties['mu'])+'\n')
    logging.info('\n'+'Start Time:   ' + timeDirectory+'\n')

def IterInfo(IterationsN):
    logging.info('|==========================================================================|')
    logging.info('                          Starting Time Loop')
    logging.info('|==========================================================================|')
    logging.info('                           Global Iter ' + str(IterationsN))
    logging.info('|--------------------------------------------------------------------------|')
    logging.info('|--------------------------------------------------------------------------|')
    logging.info('|  Equation  |     RMS     |     MAX     | initialResidual | finalResidual |')
    logging.info('|--------------------------------------------------------------------------|')

def IterTime(costTime):
    logging.info("This Iter cost %.3f minutes\n" % costTime)