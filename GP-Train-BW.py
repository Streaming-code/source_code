# -*- coding: UTF-8 -*-
import functools
import operator
import math
import random
import datetime
import time
import string
import dispy
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv


from deap.algorithms import varAnd
import sys
sys.path.append("../Compare/RobustMPC/")

import invalidBWTest



def setup(TTBWSeries):
	
	global TTTBWSeries

	TTTBWSeries=TTBWSeries
	
	return 0


	

	
def cleanup():

	global TTTBWSeries
	
	del TTTBWSeries


	
	
	
	



def protectedDiv(left, right):#自己定义的除法函数
    try:
        return left / right
    except ZeroDivisionError:
        return 1


		
		
def VrateCheck(VrateList,predictRate):  #根据计算出的predictRate，在VrateList中选择合适的速率
	Vrate=0
	if predictRate>=max(VrateList):
		Vrate=max(VrateList)
	elif predictRate<=min(VrateList):
		Vrate=min(VrateList)
	elif predictRate<max(VrateList) and predictRate>min(VrateList):
		for i in range(1,len(VrateList)):
			if predictRate>=VrateList[i-1] and predictRate<VrateList[i]:
				Vrate=VrateList[i-1]
				break
	return int(max(Vrate,min(VrateList)))
	

	

def simulatePlay(BWSeries, videoDuration, code, argument, context):
	#worker使用的库要在这里引入
	import operator	
	import math
	
		
	#将树从字符串形式变成func形式
	args = ",".join(arg for arg in argument)
	code = "lambda {args}: {code}".format(args=args, code=code)
	func=eval(code, context, {})

	
	
	
	
	VrateList=[200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]#kbps
	
	#初始变量
	startupSegNum=10#seg
	VrateTmp=VrateList[3] #初始比特率
	segmentDuration=2

	
	#跟随状态变量		
	playback=[]  #playback 是记录每一个0.1s的播放状态，0的话表示正在卡顿，1表示不卡顿
	downloadDuration=0#已经下载的segment的总时间
	buffTime=0
	bytesLeft=0
	segNum=0 
	toRecv=min(segmentDuration,videoDuration-downloadDuration)#下一个要接受的segement的长度，虽然segement长度固定，但在视频尾部可能会小于正常的segement长度
	SelectedRateList=[] #记录每一个segment的比特率
	bufferFifo=[]#存放当前存在于buffer的数据
	startupDuration=0
	rebufferDuration=0
	rebufFlag=0
	finishFlag=0
	
	startSegmentDownloadTime=0
	endSegmentDownloadTime=0
	segmentDownloadRateList=[]
	
	

	
	
	indexList = range(len(BWSeries))
	for currentTime in indexList:
		
		
		
		currentBW = BWSeries[currentTime]*8


		bytesLeft=bytesLeft+currentBW

		
			
		#下载
		if bytesLeft>=VrateTmp*toRecv: #当前接收到的数据是一个完整的segment
				

			bytesLeft=0
			buffTime=buffTime+toRecv
			SelectedRateList.extend([VrateTmp])
			downloadDuration=downloadDuration+toRecv
			segNum+=1
			for i in range(toRecv):
				bufferFifo.extend([VrateTmp])

			
			endSegmentDownloadTime = currentTime+1
			

			segmentDownloadRateList.extend([round(float(segmentDuration*VrateTmp)/(endSegmentDownloadTime-startSegmentDownloadTime),2)])
			
			
			if downloadDuration==videoDuration:#完成下载即可退出
				finishFlag=1
				break
			

			startSegmentDownloadTime=endSegmentDownloadTime

			
			toRecv=min(segmentDuration,videoDuration-downloadDuration)  #下一个要接受的segement片段长度
			



			
			
			#确定下一个seg的比特率
			if segNum>=startupSegNum:
			
			

				previousBW=sum(segmentDownloadRateList[-10:])/len(segmentDownloadRateList[-10:])

				VrateTmp=func(float(previousBW), buffTime, float(VrateTmp), (videoDuration-downloadDuration))						
				VrateTmp= VrateCheck(VrateList,VrateTmp)				
						
			
				
		
		
		#播放
		if(segNum<startupSegNum):#处在初始缓冲中
			
				
			playback.extend([0])
			startupDuration=startupDuration+1	
			
			
		else:#初始缓冲已经结束
				

			if buffTime>=1:
				playback.extend([1])
				buffTime=buffTime-1	
				bufferFifo.pop(0)	
			
			else:

				playback.extend([0])
				rebufferDuration=rebufferDuration+1	
				rebufFlag=1
				
				
				
				
	
	
	#输出统计变量

	utility=-9999999		

	
	if finishFlag==1:
	
		
		
		averageBitrate=float(sum(SelectedRateList))/segNum	
		qualityVar=sum([math.fabs(SelectedRateList[i]-SelectedRateList[i-1]) for i in range(1,len(SelectedRateList))])			
		utility=(averageBitrate-qualityVar/(segNum)-3000*float(rebufferDuration)/segNum) #原始QOE
	


	return finishFlag, utility

	


	


def SlidingTest(code, argument, context, videoDuration):

	

	global TTTBWSeries


	
	QOEList=[]
	timepoint=0
	
	for timepoint in range(0,len(TTTBWSeries),2*videoDuration):	



		#开始实验	
		BWSeries=TTTBWSeries[timepoint:timepoint+2*int(videoDuration)]	
		
		finishFlag, utility = simulatePlay(BWSeries, videoDuration, code, argument, context)
		
		QOEList.extend([utility])

		
	
	if len(QOEList)==0:
		return 0
	else:	
		return sum(QOEList)/len(QOEList)
	





	
	
def evaluate(population,cluster):#找出种群中没有被评估的个体依次评估	

	# 找出没有被评估的个体
	invalid_ind = [ind for ind in population if not ind.fitness.valid]

	
	#使用dispy的分布式评估
	jobs = []
	
	for ind in invalid_ind:#找出没有被评估的树，依次评估
	
		code = str(ind)#将树转化为字符串形式	
	
		job = cluster.submit(code, argument, context, videoDuration)
		
		jobs.append(job)
			

	for ind, job in zip(invalid_ind, jobs):
		
		utility = job()	
	
		fit=(utility,)
		
		ind.fitness.values = fit
		

	'''

	
	for ind in invalid_ind:#串行调试程序
		
		code = str(ind)	
	
		utility = SlidingTest(code, argument, context,TTBWSeries,videoDuration)	
		
	
		fit=(utility, )
		
		ind.fitness.values = fit
	'''	
	
	return population
	
	

	
	
	
	
	
def printBest(population,gen):
	
	maxInt=-9999999
	specialInd=''

	
	for indiv in population:
		if indiv.fitness.values[0]>maxInt:
			maxInt=indiv.fitness.values[0]
			specialInd = str(indiv)

			
	print gen,maxInt
	
	return specialInd, maxInt


		
	

def evol(cluster):



	
	#初始种群产生与评估
	pop = toolbox.population(n=200)#产生初始种群，包含n个个体	
	
	pop = evaluate(pop,cluster)
	
	printBest(pop,0)

	
	
	# Begin the generational process
	for gen in range(1, generation + 1):
	
		# 选择，第二个参数表示选择后后代的个数，第三个表示每一轮选择使用的个体数
		offspring = toolbox.select(pop, len(pop), int(len(pop)/4))

		# 交配和变异，该函数原理是：种群个体遵循一定的概率发生交配和变异，如果发生，父母个体被子女个体替换，维持种群数目不变
		offspring = varAnd(offspring, toolbox, mateProb, mutateProb)

		offspring = evaluate(offspring,cluster)

		pop[:] = offspring
		
		printBest(pop,gen)	
		

	return printBest(pop,generation)
	
	
	



def main():

	f=open('./trainingData/'+str(sessionNum)+' session '+str(generation)+' gen Train.txt','w')




	for throughputLevel in range(0,10):
	
		print "Level", throughputLevel

		count=0

		TTBWSeries=[]

		while(True):#select enough sessions
			
			timepoint=random.randint(videoDuration,len(TBWSeries)-1) #random generate session starting point
			
			BWSeries=TBWSeries[timepoint:timepoint+2*videoDuration]

			finishFlag = invalidBWTest.simulatePlay(BWSeries,videoDuration,TrainedOptimal)

			if finishFlag==1:

				testedList=TBWSeries[max(timepoint-videoDuration,0):timepoint]
				testedLevel = int(sum(testedList)/len(testedList)/125)

				if testedLevel>9:
					testedLevel=9
				
				if(testedLevel==throughputLevel):
					TTBWSeries.extend(BWSeries)
					count+=1

			if count>=sessionNum:
				break




		for num in range(len(seedList)):
			
			print "num",num,"count",count	

			random.seed(seedList[num])
		

			cluster = dispy.JobCluster(SlidingTest, nodes=nodes, setup=functools.partial(setup, TTBWSeries), cleanup=cleanup, port="8888", ip_addr='192.168.80.133', depends=[protectedDiv, VrateCheck, simulatePlay])# master's IP
	
			specialInd, maxInt = evol(cluster)#进行演化


			cluster.close()


			f.write(str(throughputLevel)+'\t'+str(maxInt)+'\t'+specialInd+"\t"+"\n")
		

			f.flush()


	f.close()

	
	
	





	

	

##################Tracedata初始化####################

generation=int(raw_input('Generation Num: '))
sessionNum=int(raw_input('Session Num: '))

print 'Test is ongoing.....'



videoDuration=300



TBWSeries=[]


fr=open('../../../TraceData/Combined-Trace.txt','r')
for line in fr:

		
	TBWSeries.extend([float(line.split("\n")[0])])

	
fr.close()



TrainedOptimal=invalidBWTest.init()




###############Deap 初始化####################

seedList=[50, 477, 6222, 87878]

#创建集合类
pset = gp.PrimitiveSet("MAIN", 4)#2个自变量 

#函数
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)


#常量和变量
pset.addEphemeralConstant("constant", lambda: random.uniform(-10, 10))

pset.renameArguments(ARG0='throughput')
pset.renameArguments(ARG1='bufferOccuy')
pset.renameArguments(ARG2='lastBitrate')
pset.renameArguments(ARG3='durationLeft')

#以字符串的形式提取出自变量和函数
argument=pset.arguments
context=pset.context


#创建个体类，并告知fitness越大性能越好
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

#创建函数的工具类
toolbox = base.Toolbox()



#初始化种群函数
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#算子函数，选择，交配，变异
toolbox.register("select", tools.selTournament)
toolbox.register("mate", gp.cxOnePoint)#交配
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=5)#变异和变异控制参数
toolbox.register("mutate", gp.mutInsert, pset=pset)


mateProb = 0.9 #交配概率
mutateProb = 0.8 #变异概率



#膨胀控制，防止交配和变异后树的高度太大引起内存不足
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))



#############################dispy初始化####################################


nodes = ['192.168.80.181','192.168.80.182','192.168.80.183','192.168.80.184','192.168.80.185','192.168.80.186','192.168.80.191', '192.168.80.192', '192.168.80.193', '192.168.80.194', '192.168.80.195', '192.168.80.196', '192.168.80.197', '192.168.80.198', '192.168.80.199', '192.168.80.200', '192.168.80.201', '192.168.80.202', '192.168.80.203', '192.168.80.204', '192.168.80.205', '192.168.80.206', '192.168.80.207']    # 配置worker的IP

#nodes = ['192.168.80.181','192.168.80.182','192.168.80.183','192.168.80.184','192.168.80.185','192.168.80.186']


	
main()
	
	

	
