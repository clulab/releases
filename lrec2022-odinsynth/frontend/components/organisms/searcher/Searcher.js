import React, { useState } from 'react';
import {
  Box,
  Paper,
  List,
  CardContent,
  Card,
  Button,
  TextField,
  ListItem,
  CircularProgress
} from '@mui/material';



import {
  //Remove as RemoveIcon,
  Search as SearchIcon,
  //Add as AddIcon
} from '@mui/icons-material';

import { Annotator, AnnotatorContext } from '../../'


export function SearcherContext(props) {
  return (
    <>
      <Box m={2} mt={0}>
        <SearchBar resultButtonClicked={props.onClickSearch} />
      </Box>
      <Box m={2}>
        <SearchResultContext
          onClickAnnotation={props.onClickAnnotation}
          results={props.searchResults}
          isLoading={props.isLoading}
        />
      </Box>
    </>
  )
}





export function Searcher(props) {
  return (
    <>
      <Box m={2} mt={0}>
        <SearchBar resultButtonClicked={props.onClickSearch} />
      </Box>
      <Box m={2}>
        <SearchResult
          onClickAnnotation={props.onClickAnnotation}
          results={props.searchResults}
          isLoading={props.isLoading}
        />
      </Box>
    </>
  )
}


// TODO
function SearchBar(props) {
  //
  const [searchString, setSearchString] = useState('')
  //
  function handleClickSearch() {
    props.resultButtonClicked(searchString)
  }
  //
  function handleChangeSearchString(event) {
    setSearchString(event.target.value)
  }
  //
  return (
    <Paper>
      <Box
        p={2}
        display="flex"
        flexDirection="row"
        justifyContent="center"
        alignItems="center"
      >
        <Box flex={1}>
          <TextField
            onChange={handleChangeSearchString}
            variant="outlined"
            placeholder="Query"
            fullWidth
          />
        </Box>
        <Box>
          <Button size="large" onClick={handleClickSearch} >
            <SearchIcon /> Search
          </Button>
        </Box>
      </Box>
    </Paper>
  );
}

// Search Result component
// Will contain the annotator
function SearchResult(props) {
  let results = 'Empty...'
  if (props.results) {
    results = props.results.map(
      (result) =>
        <ListItem divider>
          <Annotator
            text={result.text}
            docId={result.docId}
            sentId={result.sentId}
            annotator
            onClickAnnotation={props.onClickAnnotation}
            showCommands={true}
          />
        </ListItem>
    )
  }
  return (
    <Card>
      <CardContent>
        {props.isLoading &&
          <CircularProgress />
        }
        <List>
          {results}
        </List>
      </CardContent>
    </Card>
  );
}

function SearchResultContext(props) {
  let results = 'Empty...'
  if (props.results) {
    results = props.results.map(
      (result, k) =>
        <ListItem  key={k} divider>
          {/* TODO: we need another anotator that does what we need */}
          <AnnotatorContext
            text={result.text}
            docId={result.docId}
            sentId={result.sentId}
            annotator
            onClickAnnotation={props.onClickAnnotation}
            showCommands={true}
          />
        </ListItem>
    )
  }
  return (
    <Card>
      <CardContent>
        {props.isLoading &&
          <CircularProgress />
        }
        <List>
          {results}
        </List>
      </CardContent>
    </Card>
  );
}
