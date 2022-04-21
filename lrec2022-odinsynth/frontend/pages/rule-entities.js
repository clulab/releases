import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useRouter } from 'next/router'
import { Annotator, Searcher, Stash } from '../components/'

import {
  Paper,
  Grid,
  Box,
  Card,
  Typography,
  CardContent,
  List,
  ListItem,
  Button,
  CardActions,
  CircularProgress,
} from '@mui/material';

import {
  Backspace as BackspaceIcon,
  Forward as ForwardIcon,
  Block as BlockIcon,
  Save as SaveIcon,
  Edit as EditIcon,
} from '@mui/icons-material';


function ExtractionComponent(props) {
  const [showButton, setShowButton] = useState(false)

  const addAnnotations = (text, matchStart, matchEnd) => {
    let tokens = text.split(' ')
    let annotatedText = tokens.slice(matchStart, matchEnd).join(' ')
    let left = tokens.slice(0, matchStart).join(' ') + ' '
    let right = ' ' + tokens.slice(matchEnd).join(' ')
    return (
      <Box>
        {left}
        <Button variant='outlined'>
          {annotatedText}
        </Button>
        {right}
      </Box>
    )
  }

  return (
    <ListItem divider
      onMouseEnter={() => setShowButton(true)}
      onMouseLeave={() => setShowButton(false)}
      onClick={() => props.sendToStash(props.extraction)}
    >
      <Box display="flex" justifyContent="center" alignItems="center">
        <Box flexGrow={1}>
          <Typography variant="body1">
            {addAnnotations(props.extraction.text, props.extraction.matchStart, props.extraction.matchEnd)}
          </Typography>
        </Box>
        <Box>
          {showButton && (
            <Button color='secondary' variant='contained'>
              <BlockIcon />
            </Button>
          )}
        </Box>
      </Box>
    </ListItem>
  )

}

//
function ResultExtractions(props) {
  // add matches
  // get results items
  let results = props.results.map(t =>
  (
    <ExtractionComponent sendToStash={props.sendToStash} extraction={t} />
  )
  )
  //
  return (
    <Card>
      <Box p={2}>
        <Typography variant="h2">Extractions</Typography>
      </Box>
      <CardContent>
        <List>
          {results}
        </List>
      </CardContent>
    </Card>
  )
}

function RuleDashboard(props) {
  return (
    <Card>
      <CardContent>
        <Typography gutterBottom variant="h4">
          Rule generated:
        </Typography>
        <Typography variant="h5" gutterBottom>
          {props.rule}
          <Button color='secondary' onClick={() => { alert('please code me') }}  ><EditIcon /></Button>
        </Typography>
        <Typography gutterBottom variant="body1" gutterBottom>
          Number of states: {props.ruleInfo.nSteps}<br />
          Number of steps: {props.ruleInfo.currentSteps}<br />
        </Typography>
      </CardContent>
      <CardActions>
        <
          Button variant="contained" color="secondary">
          Go back to stash &nbsp;
          <BackspaceIcon />
        </Button>
        <Button variant="contained" color="primary" onClick={props.onClickNextRule}>
          Next rule &nbsp;
          <ForwardIcon />
        </Button>
        {props.isLoading &&
          <CircularProgress />
        }
      </CardActions>
    </Card>
  )
}
// Base class AnnotationApp
// Children: Search and Stash
export default function RuleGenerationApp() {
  // state
  const [rule, setRule] = useState("Generating rule...")
  const [results, setResults] = useState([])
  const [isLoading, setLoading] = useState(false)
  const [stash, setStash] = useState([])
  const [ruleInfo, setRuleInfo] = useState({ nSteps: 0, currentSteps: 0 })

  // TODO get inittial data here
  // check if there is data in geto
  // if there is no data in get

  const router = useRouter()
  useEffect(() => {
    let keys = Object.keys(router.query)
    if (keys.length > 0) {
      let qReturn = JSON.parse('{"data":' + keys + '}')
      setStash(qReturn.data)
      getFirstRule(qReturn.data)
    } else {
      setRule('Stash is empty.')
    }
    // TODO: if no data in, what do we do?
  }, [])

  // nextRule button
  const getNextRule = () => {
    // TODO: add a spinner to the extractions part
    // TODO: don't allow the user to click next before the last action is canceled

    setLoading(true)
    axios.get(process.env.apiUri + 'nextRule')
      .then((response) => {
        let [rule, nSteps, currentSteps] = response.data.split('\t')
        console.log(response)
        setRuleInfo({ nSteps: nSteps, currentSteps: currentSteps })
        setRule(rule)
        updateResults(rule)
      })
  }

  // update results
  const updateResults = (query) => {
    axios.get(process.env.apiUri + 'search?query=' + encodeURIComponent(query))
      .then((response) => {
        if (response.data) {
          setResults(response.data)
        }
        // FIXME is this correct? Handling when no rule was found
        setLoading(false)
      })
  }

  // TODO: getFirstRule
  const getFirstRule = (data) => {
    setLoading(true)
    let payload = { 'data': data }
    // console.log(payload)

    axios.post(process.env.apiUri + 'generateRule', payload)
      .then((response) => {
        let [rule, nSteps, currentSteps] = response.data.split('\t')
        setRule(rule)
        setRuleInfo({ nSteps: nSteps, currentSteps: currentSteps })
        updateResults(rule)
      })
  }

  // function to add stuff to the stash
  const sendToStash = (value) => {
    let newStash = stash.concat(value)
    console.log(newStash)
    setStash(newStash)
    getFirstRule(newStash)
  }
  const removeFromStash = (value) => {
    let newStash = stash.filter((v) => v != value)
    // shoul count only positives
    if (newStash.filter((v) => v.span != null).length == 0) {
      // stash is empty, redirect to annotation
      history.push('/')
    }
    console.log(newStash)
    setStash(newStash)
    getFirstRule(newStash)
  }
  return (
    <div style={{ margin: 24 }}>
      <Grid
        spacing={2}
        direction="row"
        justify="center"
        alignItems="flex-start"
        container
      >
        <Grid xs={12} sm={6} item>
          <Box>
            <Stash annotations={stash} hideButton={true} removeFromStash={removeFromStash} />
          </Box>
          <Box mt={2}>
            <RuleDashboard rule={rule} ruleInfo={ruleInfo} isLoading={isLoading} onClickNextRule={getNextRule} />
          </Box>
          <Box mt={2}>
            <Paper>
              <Box p={1}>
                <Button onClick={() => {
                  console.log(stash)
                  console.log(rule)
                  console.log(results)
                }} variant='contained'>Download data<SaveIcon /></Button>
              </Box>
            </Paper>
          </Box>
        </Grid >
        <Grid xs={12} sm={6} item>
          <Box>
            <ResultExtractions results={results} sendToStash={sendToStash} />
          </Box>
        </Grid>
      </Grid>
      <Box mt={2}>
        {/*
          *
          <Copyright />
          *
          */}
      </Box>
    </div>
  );
}
