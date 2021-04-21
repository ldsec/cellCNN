package main_test

import (
	"fmt"
	timedataunlynx "github.com/ldsec/unlynx/simul/test_data/time_data"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

var flags = []string{"bf", "depth", "rounds", "runwait", "hosts",
	"DecentralizedCNN(SIMULATION)", "CnnProtocolSimul(PRE-COMPUTATION)",
	"LocalIteration", "UpdateWeights", "Combine",
}

func TestReadTimeData(t *testing.T) {
	//_, err := timedataunlynx.ReadTomlSetup("config-test.toml", 0)
	var f *os.File
	var err error
	f, err = os.Create("../eval/time-data.csv")
	defer f.Close()
	f.WriteString("bf,depth,rounds,runwait,hosts,DecentralizedCNN(SIMULATION),CnnProtocolSimul(PRE-COMPUTATION),LocalIteration,UpdateWeights,Combine\n")
	nHosts := []string{"1", "3", "5", "7", "9", "10", "15", "20", "25", "30", "35", "40"}
	for i := range nHosts {
		filename := "test_data/time" + nHosts[i] + ".csv"
		m, _ := timedataunlynx.ReadDataFromCSVFile(filename, flags)
		fmt.Println(m)
		for j := range flags {
			flag := flags[j]
			if j > 0 {
				f.WriteString(",")
			}
			f.WriteString(m[flag][0])
		}
		f.WriteString("\n")
		//timedataunlynx.WriteDataFromCSVFile("time-data.txt", flags, m, i, nil)
	}
	require.NoError(t, err)
}

func TestReadLocal(t *testing.T) {
	//_, err := timedataunlynx.ReadTomlSetup("config-test.toml", 0)
	var err error
	filename := "test_data/config-test.csv"
	m, _ := timedataunlynx.ReadDataFromCSVFile(filename, flags)
	fmt.Println(m)
	for j := range flags {
		flag := flags[j]
		fmt.Println(flag, m[flag][0])
	}
	require.NoError(t, err)
}
