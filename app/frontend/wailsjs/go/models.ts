export namespace domain {
	
	export class AreaInfo {
	    id: string;
	    label: string;
	    blurb: string;
	
	    static createFrom(source: any = {}) {
	        return new AreaInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.label = source["label"];
	        this.blurb = source["blurb"];
	    }
	}
	export class AreaStat {
	    area: string;
	    label: string;
	    average: number;
	    samples: number;
	    trend: number;
	    last: number;
	
	    static createFrom(source: any = {}) {
	        return new AreaStat(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.area = source["area"];
	        this.label = source["label"];
	        this.average = source["average"];
	        this.samples = source["samples"];
	        this.trend = source["trend"];
	        this.last = source["last"];
	    }
	}
	export class BoardEdge {
	    id: string;
	    source: string;
	    target: string;
	    label: string;
	
	    static createFrom(source: any = {}) {
	        return new BoardEdge(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.source = source["source"];
	        this.target = source["target"];
	        this.label = source["label"];
	    }
	}
	export class BoardNode {
	    id: string;
	    kind: string;
	    label: string;
	    detail: string;
	    x: number;
	    y: number;
	
	    static createFrom(source: any = {}) {
	        return new BoardNode(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.kind = source["kind"];
	        this.label = source["label"];
	        this.detail = source["detail"];
	        this.x = source["x"];
	        this.y = source["y"];
	    }
	}
	export class Criterion {
	    id: string;
	    area: string;
	    title: string;
	    detail: string;
	    weight: number;
	
	    static createFrom(source: any = {}) {
	        return new Criterion(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.area = source["area"];
	        this.title = source["title"];
	        this.detail = source["detail"];
	        this.weight = source["weight"];
	    }
	}
	export class CriterionScore {
	    criterionId: string;
	    area: string;
	    title: string;
	    score: number;
	    evidence: string;
	
	    static createFrom(source: any = {}) {
	        return new CriterionScore(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.criterionId = source["criterionId"];
	        this.area = source["area"];
	        this.title = source["title"];
	        this.score = source["score"];
	        this.evidence = source["evidence"];
	    }
	}
	export class Curveball {
	    atPct: number;
	    title: string;
	    body: string;
	    fired: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Curveball(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.atPct = source["atPct"];
	        this.title = source["title"];
	        this.body = source["body"];
	        this.fired = source["fired"];
	    }
	}
	export class Event {
	    id: string;
	    // Go type: time
	    at: any;
	    elapsedSec: number;
	    kind: string;
	    severity: string;
	    title: string;
	    body: string;
	    areas: string[];
	
	    static createFrom(source: any = {}) {
	        return new Event(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.at = this.convertValues(source["at"], null);
	        this.elapsedSec = source["elapsedSec"];
	        this.kind = source["kind"];
	        this.severity = source["severity"];
	        this.title = source["title"];
	        this.body = source["body"];
	        this.areas = source["areas"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Problem {
	    id: string;
	    mode: string;
	    level: string;
	    title: string;
	    statement: string;
	    requirements: string[];
	    constraints: string[];
	    rubric: Criterion[];
	    curveballs: Curveball[];
	    referenceOutline: string[];
	    starter: string;
	    language: string;
	
	    static createFrom(source: any = {}) {
	        return new Problem(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.mode = source["mode"];
	        this.level = source["level"];
	        this.title = source["title"];
	        this.statement = source["statement"];
	        this.requirements = source["requirements"];
	        this.constraints = source["constraints"];
	        this.rubric = this.convertValues(source["rubric"], Criterion);
	        this.curveballs = this.convertValues(source["curveballs"], Curveball);
	        this.referenceOutline = source["referenceOutline"];
	        this.starter = source["starter"];
	        this.language = source["language"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Profile {
	    mode: string;
	    sessions: number;
	    totalMinutes: number;
	    averageScore: number;
	    areas: AreaStat[];
	    strongest: string[];
	    weakest: string[];
	
	    static createFrom(source: any = {}) {
	        return new Profile(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.mode = source["mode"];
	        this.sessions = source["sessions"];
	        this.totalMinutes = source["totalMinutes"];
	        this.averageScore = source["averageScore"];
	        this.areas = this.convertValues(source["areas"], AreaStat);
	        this.strongest = source["strongest"];
	        this.weakest = source["weakest"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Review {
	    overall: number;
	    verdict: string;
	    summary: string;
	    scores: CriterionScore[];
	    strengths: string[];
	    gaps: string[];
	    nextSteps: string[];
	    missedOutline: string[];
	
	    static createFrom(source: any = {}) {
	        return new Review(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.overall = source["overall"];
	        this.verdict = source["verdict"];
	        this.summary = source["summary"];
	        this.scores = this.convertValues(source["scores"], CriterionScore);
	        this.strengths = source["strengths"];
	        this.gaps = source["gaps"];
	        this.nextSteps = source["nextSteps"];
	        this.missedOutline = source["missedOutline"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Snapshot {
	    nodes: BoardNode[];
	    edges: BoardEdge[];
	    notes: string;
	    code: string;
	    language: string;
	
	    static createFrom(source: any = {}) {
	        return new Snapshot(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.nodes = this.convertValues(source["nodes"], BoardNode);
	        this.edges = this.convertValues(source["edges"], BoardEdge);
	        this.notes = source["notes"];
	        this.code = source["code"];
	        this.language = source["language"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class SessionSpec {
	    mode: string;
	    level: string;
	    topic: string;
	    language: string;
	    durationSec: number;
	    customStatement: string;
	    coachEnabled: boolean;
	    coachIntervalSec: number;
	
	    static createFrom(source: any = {}) {
	        return new SessionSpec(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.mode = source["mode"];
	        this.level = source["level"];
	        this.topic = source["topic"];
	        this.language = source["language"];
	        this.durationSec = source["durationSec"];
	        this.customStatement = source["customStatement"];
	        this.coachEnabled = source["coachEnabled"];
	        this.coachIntervalSec = source["coachIntervalSec"];
	    }
	}
	export class Session {
	    id: string;
	    spec: SessionSpec;
	    problem: Problem;
	    // Go type: time
	    startedAt: any;
	    // Go type: time
	    endedAt: any;
	    elapsedSec: number;
	    events: Event[];
	    final: Snapshot;
	    review?: Review;
	    phase: string;
	    provider: string;
	    model: string;
	
	    static createFrom(source: any = {}) {
	        return new Session(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.spec = this.convertValues(source["spec"], SessionSpec);
	        this.problem = this.convertValues(source["problem"], Problem);
	        this.startedAt = this.convertValues(source["startedAt"], null);
	        this.endedAt = this.convertValues(source["endedAt"], null);
	        this.elapsedSec = source["elapsedSec"];
	        this.events = this.convertValues(source["events"], Event);
	        this.final = this.convertValues(source["final"], Snapshot);
	        this.review = this.convertValues(source["review"], Review);
	        this.phase = source["phase"];
	        this.provider = source["provider"];
	        this.model = source["model"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	
	export class Status {
	    phase: string;
	    sessionId: string;
	    problem?: Problem;
	    elapsedSec: number;
	    remainingSec: number;
	    durationSec: number;
	    coachEnabled: boolean;
	    events: Event[];
	    coveredAreas: string[];
	    review?: Review;
	    error: string;
	
	    static createFrom(source: any = {}) {
	        return new Status(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.phase = source["phase"];
	        this.sessionId = source["sessionId"];
	        this.problem = this.convertValues(source["problem"], Problem);
	        this.elapsedSec = source["elapsedSec"];
	        this.remainingSec = source["remainingSec"];
	        this.durationSec = source["durationSec"];
	        this.coachEnabled = source["coachEnabled"];
	        this.events = this.convertValues(source["events"], Event);
	        this.coveredAreas = source["coveredAreas"];
	        this.review = this.convertValues(source["review"], Review);
	        this.error = source["error"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}

}

export namespace llm {
	
	export class Config {
	    kind: string;
	    model: string;
	    host: string;
	    apiKey: string;
	
	    static createFrom(source: any = {}) {
	        return new Config(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.kind = source["kind"];
	        this.model = source["model"];
	        this.host = source["host"];
	        this.apiKey = source["apiKey"];
	    }
	}
	export class ModelInfo {
	    name: string;
	    sizeBytes: number;
	
	    static createFrom(source: any = {}) {
	        return new ModelInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.sizeBytes = source["sizeBytes"];
	    }
	}

}

export namespace store {
	
	export class Settings {
	    llm: llm.Config;
	    defaults: domain.SessionSpec;
	
	    static createFrom(source: any = {}) {
	        return new Settings(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.llm = this.convertValues(source["llm"], llm.Config);
	        this.defaults = this.convertValues(source["defaults"], domain.SessionSpec);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}

}

