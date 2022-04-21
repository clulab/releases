import React, { useEffect, useState } from 'react';
//import {useHistory} from "react-router-dom"

import {
  ListItem,
  Box,
  Paper,
  Typography,
  List,
  Button,
} from '@mui/material';
//
import {
  Delete as DeleteIcon,
} from '@mui/icons-material';
//
import Link from "next/link"

export function Stash(props) {

  useEffect(() => {
    if(props.annotations && props.annotations.length > 0){
      window.localStorage.setItem("stash_annotations", JSON.stringify(props.annotations))
    }
  });

  // TODO: add a button with on hover to remove stuff from stash
  let annotations = 'Empty';
  if (props.annotations) {
    //console.log(JSON.stringify(props.annotations))
    annotations = props.annotations.map(
      (ann, ix) =>
        <ListItem divider  key= { ix }>
          <AnnotatorVizualizer data={ann} removeFromStash={props.removeFromStash} />
        </ListItem>
    )
  }
  //
  return (
    <Paper>
      <Box p={2}>
        <Typography variant="h2">Stash</Typography>
        {annotations.length > 0 && (
          <Box>
            <List>
              {annotations}
            </List>
            {!props.hideButton && (
              <Button variant='contained'>
              <Link
                href={{
                  pathname: '/rule-context',
                  query: JSON.stringify(props.annotations)
                }}
              >
                Generate Rule
              </Link>
              </Button>
            )}
          </Box>
        )}
        {annotations.length === 0 && (
          <Box py={2}>
            Empty...
          </Box>
        )}
      </Box>
    </Paper>
  );
}
//
function AnnotatorVizualizer(props) {
  let [showButton, setShowButton] = useState(false)
  // make tokens
  // define default `toShow`
  let toShow = (
    <Box flexGrow={1}>
      {props.data.text}
    </Box>
  )
  // default is just showing the text
  if (props.data.span) {
    // get span
    let text = props.data.text.split(' ')
    let start = props.data.span.start
    let end = props.data.span.end
    // split the text to get the annotations
    let annotatedText = text.slice(start, end).join(' ')
    let left = text.slice(0, start).join(' ') + ' '
    let right = ' ' + text.slice(end).join(' ')
    //
    toShow = (
      <Box flexGrow={1}>
        {left}
        <Button variant='outlined'>
          {annotatedText}
        </Button>
        {right}
      </Box>
    )
  }
  //
  return (
    <Box
      onMouseEnter={() => setShowButton(true)}
      onMouseLeave={() => setShowButton(false)}
      display='flex'
    >
      {toShow}
      {showButton && (
        <Box>
          <Button
            onClick={() => props.removeFromStash(props.data)}
            variant='contained'
            color='secondary'>
            <DeleteIcon />
          </Button>
        </Box>
      )}
    </Box>
  )
}
