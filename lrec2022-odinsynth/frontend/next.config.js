module.exports = {
  reactStrictMode: true,
  
  env: {
    apiUri: process.env.API_URL,
  },
  
  async headers() {

    return [
      {
        source: "/api/download-results", // This is necessary to tell the browser to download the file rather than display the json contents
        headers: [
          {
            key: "Content-Disposition",
            value: 'attachment; filename="results.json"'
          }
        ]
      }
    ]
  }
}
