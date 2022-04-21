/*
 import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'
 * */


import React, { useState } from 'react';
import { Grid, Box } from '@mui/material';
import axios from 'axios';

// todo this should also be a component
// import Copyright from './Copyright';

import { Searcher, Stash } from '../components/'

// Bade class AnnotationApp
// Children: Search and Stash
export default function AnnotationApp() {

  const [results, setResults] = useState('')
  const [stash, setStash] = useState([])
  const [isLoading, setLoading] = useState(false)

  const handleSearchResult = (result) => {
    // console.log(result)
    // console.log(process.env.apiUri + 'search?query=' + result)
    // if result not empty
    if (result) {
      axios.get(process.env.apiUri + 'search?query=' + result)
      .then((response) => {
          if (response.data) {
            console.log(response.data)
            // setting state triggers refresh
            setResults(response.data)
            setLoading(false)
          }
        })
    }
  }

  const sendToStash = (value) => {
    setStash(stash.concat(value))
  }
  const removeFromStash = (value) => {
    setStash(stash.filter((v) => v != value))
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
          <Searcher
            onClickSearch={handleSearchResult}
            onClickAnnotation={sendToStash}
            searchResults={results}
            isLoading={isLoading}
          />
        </Grid>
        <Grid xs={12} sm={6} item>
          <Stash annotations={stash} removeFromStash={removeFromStash} />
        </Grid>
      </Grid>
      <Box mt={2}>
        {/*
           <Copyright />
        */}
      </Box>
    </div >
  );
  //
}
