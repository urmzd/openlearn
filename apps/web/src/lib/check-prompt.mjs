// A reader works a question on paper, then asks an AI assistant to check the
// answer. The prompt carries the question and the book's worked answer, so the
// assistant checks against the book and not against its own guess.

const clean = (text) => String(text ?? '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();

export function checkPrompt({ book, section, url, question, answer }) {
	const source = [section && `"${clean(section)}"`, book && `in the Openlearn book "${clean(book)}"`].filter(Boolean).join(' ');
	return [
		`I am studying ${source || 'an Openlearn book'}${url ? ` (${url})` : ''}. I worked this question on paper.`,
		`Question:\n${clean(question)}`,
		'My answer:\n[Type your answer here, or attach a photo of your page.]',
		[
			'Check my answer against the reference answer below.',
			'1. Say if my answer is correct, partly correct or incorrect.',
			'2. Point to the first step where my reasoning goes wrong, if there is one.',
			'3. Name anything important that I left out.',
			'4. Ask me one follow-up question that changes the case.',
			'Do not show the full solution unless I ask for it.',
		].join('\n'),
		...(clean(answer) ? [`Reference answer from the book, for your check only:\n${clean(answer)}`] : []),
	].join('\n\n');
}
