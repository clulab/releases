

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useRouter } from 'next/router'
import { Annotator, Searcher, Stash } from '../components/'

// Base class AnnotationApp
// Children: Search and Stash
/*
 *
export default function RuleGenApp() {
  const router = useRouter()
  let stash = router.query

  return (
    <div>
      {JSON.stringify(stash)}
    </div >
  );
  //
}
 * */

import {
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
  Link,
} from '@mui/material';

import {
  Backspace as BackspaceIcon,
  Forward as ForwardIcon,
  Block as BlockIcon,
  ConnectedTvOutlined,
} from '@mui/icons-material';


function ExtractionComponent(props) {
  const [showButton, setShowButton] = useState(false)

  const addAnnotations = (ann) => {
    let text = ann.text
    let matchStart = ann.matchStart
    let matchEnd = ann.matchEnd

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
            {addAnnotations(props.extraction)}
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
  console.log(props)
  // get results items
  let results = undefined

  if (props.results.length > 0) {
    results = props.results.map(t =>
    (
      <ExtractionComponent sendToStash={props.sendToStash} extraction={t} />
    )
    )
  }
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
  const formRef = useRef()

  return (
    <Card>
      <CardContent>
        <Typography gutterBottom variant="h4">
          Rule generated:
        </Typography>
        <Typography variant="h5" gutterBottom>
          {props.rule}
        </Typography>
        <Typography gutterBottom variant="body1" gutterBottom>
          Number of states: {props.ruleInfo.nSteps}<br />
          Number of steps: {props.ruleInfo.currentSteps}<br />
        </Typography>
      </CardContent>
      <CardActions>
        <Link href='/context'>
        <Button variant="contained" color="secondary">
          Go back to stash &nbsp;
          <BackspaceIcon />
        </Button>
        </Link>
        <Button variant="contained" color="primary" onClick={props.onClickNextRule}>
          Next rule &nbsp;
          <ForwardIcon />
        </Button>
        {/*  Download button */}
        {!props.isLoading && (
          <>
            <form ref={formRef} method='POST' action='/api/download-results' target='_blank'>
              <input type="hidden" name="ruleInfo" value={ JSON.stringify(props.ruleInfo) }></input>
              <input type="hidden" name="rule" value={ props.rule }></input>
              <input type="hidden" name="stash" value={ JSON.stringify(props.annotations.map(a => a.text)) }></input>
              <input type="hidden" name="results" value={ props.results?JSON.stringify(props.results.map(r => r.text)):""  }></input>
              <input type="hidden" name="query" value={ JSON.stringify(props.annotations) }></input>
              <input type="hidden" name="totalHits" value={  props.results?JSON.stringify(props.results.map(r => r.totalHits)):0 }></input>
            </form>
            <Button variant="contained" color="secondary" onClick={
              () => {
                formRef.current.submit()
              }
            }>
              Download
            </Button>
          </> )
        }
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
    // let keys = Object.keys(router.query)
    let keys = [window.localStorage.getItem("stash_annotations")]
    if (keys && keys.length > 0) {
      keys = JSON.parse(keys)
      setStash(keys) 
      getFirstRule(keys)
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
          setLoading(false)
        }
      })
  }

  // TODO: getFirstRule
  const getFirstRule = (data) => {
    setLoading(true)
    let payload = { 'data': data }

    axios.post(process.env.apiUri + 'generate-rule/entity/masked', payload)
      .then((response) => {
        let [rule, nSteps, currentSteps] = response.data.split('\t')
        setRule(rule)
        setRuleInfo({ nSteps: nSteps, currentSteps: currentSteps })
        updateResults(rule)
      })
  }

  /*
  // set innitial data
  let history = useHistory();
  useEffect(() => {
    // check if the prop received the data
    if (location.data) {
      // assemble the object that will be sent to the backend
      // put data on stash
      setStash(location.data)
      // load the first rule
      getFirstRule(location.data)
      // if no data was received, redirect it back to the annotation
    } else {
      history.push('/')
    }
  }, [])
   * */

  // function to add stuff to the stash
  const sendToStash = (value) => {
    let newStash = stash.concat(value)
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
            <RuleDashboard rule={rule} results={results} annotations={stash} ruleInfo={ruleInfo} isLoading={isLoading} onClickNextRule={getNextRule} />
          </Box>
        </Grid>
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
