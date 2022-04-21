
export default function downloadResults(req, res) {
	const body = req.body
	const data = {
		ruleInfo: JSON.parse(body.ruleInfo),
		rule: body.rule,
		stash: JSON.parse(body.stash),
		results: JSON.parse(body.results),
		query: JSON.parse(body.query),
		totalHits: JSON.parse(body.totalHits)
	}
	res.status(200).send(JSON.stringify(data, undefined, 2))
}