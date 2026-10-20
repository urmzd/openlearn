// Keep discovery alphabetical and teaching order faithful to each publication.
export function courseOutlines(publications, isPublished = () => true) {
	return [...publications]
		.sort((a, b) => a.manifest.title.localeCompare(b.manifest.title, 'en', { sensitivity: 'base' }))
		.map(({ manifest: course, reading }) => {
			const introductions = new Set(course.start.map(({ file }) => file));
			const referenceDirs = (course.reference ?? []).map(({ dir }) => `${dir}/`);
			const chapters = [];
			const references = [];
			for (const section of reading) {
				const main = [];
				const auxiliary = [];
				for (const file of section.files.filter(isPublished)) {
					const isReference = course.guided
						? !introductions.has(file)
						: referenceDirs.some((dir) => file.startsWith(dir)) && !introductions.has(file);
					(isReference ? auxiliary : main).push(file);
				}
				const { title, number, kind, labels = {} } = section;
				if (main.length) chapters.push({ title, number, kind, labels, files: main });
				if (auxiliary.length) references.push({ title, files: auxiliary });
			}
			return { course, chapters, references };
		});
}

export function chapterNeighbors(outlines, linkOf) {
	const owners = new Map();
	for (const { course, chapters } of outlines) {
		const files = chapters.flatMap(({ files }) => files);
		const neighbor = (file) => file ? { link: linkOf(file) } : false;
		for (const [i, file] of files.entries()) {
			const routes = owners.get(file) ?? [];
			routes.push({
				course,
				prev: i ? neighbor(files[i - 1]) : { label: 'All books', link: '/' },
				next: neighbor(files[i + 1]),
			});
			owners.set(file, routes);
		}
	}
	return owners;
}
