#!/usr/bin/env bash
sbt -mem 10000 'runMain org.clulab.clint.BuildLexicons' 'runMain org.clulab.clint.BuildEntityPatternsDump' 'runMain org.clulab.clint.BuildPatternLexicon' 'runMain org.clulab.clint.BuildEntityToPatternsIndex' 'runMain org.clulab.clint.BuildPatternToEntitiesIndex' 'runMain org.clulab.clint.CreateFilteredDataset'
