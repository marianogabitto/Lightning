

# C++ sources
CXXSRC        += Timer.cc
CXXSRC        += CounterIntEvent.cc
CXXSRC        += PointBase.cc
CXXSRC        += Point.cc
CXXSRC        += Center.cc
CXXSRC        += QTreeParam.cc
CXXSRC        += QTreeNode.cc
CXXSRC        += QTree.cc
CXXSRC        += KDTree.cc
#CXXSRC        += KMeans.cc
CXXSRC        += Light.cc


CXXOBJ        = $(CXXSRC:%.cc=%.o)

# Compilation flags
CXXFLAGS      =    -Wall -g
CXXFLAGS      +=  -O3

all:  dep test2


test2: $(CXXOBJ) test2.o
	$(CXX) -o test2 test2.o $(CXXOBJ)

test1: $(CXXOBJ) test1.o
	$(CXX) -o test1 test1.o $(CXXOBJ)


kmeans_test: $(CXXOBJ) kmeans_test.o
	$(CXX) -o kmeans_test kmeans_test.o $(CXXOBJ)

light_test: $(CXXOBJ) light_test.o
	$(CXX) -o light_test light_test.o $(CXXOBJ)

clean:
	rm -f *.o test1 kmeans_test light_test


dep: .depend

.depend: $(CXXSRC)
	$(CXX) -MM *.cc > .depend


%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $*.cc


echo:
	@echo "CXXSRC: $(CXXSRC)"
	@echo "CXXOBJ: $(CXXOBJ)"


-include .depend

