"""Hindley-Milner type inference (Algorithm W) from scratch.

This is the math made executable. Languages in the ML family (OCaml, Haskell,
Rust's local inference, even TypeScript's `infer`) all descend from one idea:
given an untyped lambda term, *compute* its most general (principal) type with no
annotations, using only two operations -- unification and let-generalization.

The term language is the simply-typed lambda calculus plus `let` (the thing that
makes polymorphism work):

    e ::= x                      -- variable
        | \\x. e                 -- abstraction (lambda)
        | e e                    -- application
        | let x = e in e         -- let-binding (generalization point)
        | <int literal>          -- a base value, type `int`

Types split into *monotypes* (tau) and *type schemes* (sigma):

    tau   ::= a | int | bool | tau -> tau          -- monotypes
    sigma ::= tau | forall a. sigma                -- schemes (polymorphic types)

The whole algorithm is four ideas:
  1. unify(t1, t2): make two monotypes equal by solving for type variables
     (with the *occurs check* to stay finite).
  2. instantiate(sigma): replace a scheme's bound vars with fresh ones, so each
     use of a polymorphic value gets its own copy ("let id = \\x.x" can be used
     at int AND at bool in the same program).
  3. generalize(env, tau): close a monotype over the variables that are free in
     it but not in the environment -- this is what turns a lambda's inferred
     monotype into a reusable forall scheme at a `let`.
  4. W(env, e): walk the term, threading a substitution, emitting fresh vars at
     lambdas and unifying at applications.

Run it:  python hindley_milner.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

# --------------------------------------------------------------------------- #
# Types: the monotype tau and the scheme sigma = forall [vars]. tau            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TVar:
    name: str  # a type variable, e.g. "t0"


@dataclass(frozen=True)
class TCon:
    name: str  # a nullary constructor, e.g. "int", "bool"


@dataclass(frozen=True)
class TArrow:
    arg: "Type"
    ret: "Type"  # the function type arg -> ret


Type = TVar | TCon | TArrow


@dataclass(frozen=True)
class Scheme:
    # forall `quantified`. `body`. An empty quantifier list is a plain monotype.
    quantified: tuple[str, ...]
    body: Type


TInt = TCon("int")
TBool = TCon("bool")

_fresh = count()


def fresh_var() -> TVar:
    return TVar(f"t{next(_fresh)}")


# --------------------------------------------------------------------------- #
# Terms: the untyped lambda calculus + let + int literals                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Lam:
    param: str
    body: "Term"


@dataclass(frozen=True)
class App:
    fn: "Term"
    arg: "Term"


@dataclass(frozen=True)
class Let:
    name: str
    value: "Term"
    body: "Term"


@dataclass(frozen=True)
class Lit:
    value: int


Term = Var | Lam | App | Let | Lit


# --------------------------------------------------------------------------- #
# Substitutions: maps from type-variable names to monotypes                   #
# --------------------------------------------------------------------------- #

Subst = dict[str, Type]


def apply(sub: Subst, t: Type) -> Type:
    """Apply a substitution to a type (replace solved variables, recursively)."""
    if isinstance(t, TVar):
        # Chase the chain: a var may map to another var that is itself solved.
        return apply(sub, sub[t.name]) if t.name in sub else t
    if isinstance(t, TArrow):
        return TArrow(apply(sub, t.arg), apply(sub, t.ret))
    return t  # TCon


def apply_scheme(sub: Subst, s: Scheme) -> Scheme:
    # Never substitute a scheme's own bound variables -- shadow them out first.
    inner = {k: v for k, v in sub.items() if k not in s.quantified}
    return Scheme(s.quantified, apply(inner, s.body))


def compose(s2: Subst, s1: Subst) -> Subst:
    """Apply s1 then s2: the standard substitution composition (s2 . s1)."""
    out: Subst = {k: apply(s2, v) for k, v in s1.items()}
    out.update(s2)
    return out


def free_vars(t: Type) -> set[str]:
    if isinstance(t, TVar):
        return {t.name}
    if isinstance(t, TArrow):
        return free_vars(t.arg) | free_vars(t.ret)
    return set()


def free_vars_scheme(s: Scheme) -> set[str]:
    return free_vars(s.body) - set(s.quantified)


# --------------------------------------------------------------------------- #
# Unification: solve t1 = t2 for the type variables (with the occurs check)   #
# --------------------------------------------------------------------------- #


class TypeError_(Exception):
    pass


def occurs(name: str, t: Type) -> bool:
    # Does `name` appear inside t? If so, unifying name=t would build an
    # infinite type (a = a -> b). The occurs check rejects exactly that.
    return name in free_vars(t)


def unify(t1: Type, t2: Type) -> Subst:
    if isinstance(t1, TVar):
        return bind(t1.name, t2)
    if isinstance(t2, TVar):
        return bind(t2.name, t1)
    if isinstance(t1, TArrow) and isinstance(t2, TArrow):
        s1 = unify(t1.arg, t2.arg)
        s2 = unify(apply(s1, t1.ret), apply(s1, t2.ret))
        return compose(s2, s1)
    if isinstance(t1, TCon) and isinstance(t2, TCon) and t1.name == t2.name:
        return {}
    raise TypeError_(f"cannot unify {show(t1)} with {show(t2)}")


def bind(name: str, t: Type) -> Subst:
    if isinstance(t, TVar) and t.name == name:
        return {}  # binding a var to itself is a no-op
    if occurs(name, t):
        raise TypeError_(f"occurs check: {name} occurs in {show(t)}")
    return {name: t}


# --------------------------------------------------------------------------- #
# instantiate / generalize: the polymorphism mechanism                         #
# --------------------------------------------------------------------------- #


def instantiate(s: Scheme) -> Type:
    """Give each use of a scheme a fresh copy of its quantified variables."""
    mapping: Subst = {q: fresh_var() for q in s.quantified}
    return apply(mapping, s.body)


@dataclass
class TypeEnv:
    # Maps program variables to type schemes (gamma in the textbooks).
    env: dict[str, Scheme] = field(default_factory=dict)

    def extend(self, name: str, scheme: Scheme) -> "TypeEnv":
        merged = dict(self.env)
        merged[name] = scheme
        return TypeEnv(merged)

    def free_vars(self) -> set[str]:
        out: set[str] = set()
        for s in self.env.values():
            out |= free_vars_scheme(s)
        return out


def generalize(env: TypeEnv, t: Type) -> Scheme:
    """Close `t` over the vars free in it but NOT bound in the environment.

    This is the heart of let-polymorphism: a lambda parameter stays monomorphic
    (its var is free in the env), but a `let`-bound value is generalized, so the
    name can be reused at many types.
    """
    quantified = tuple(sorted(free_vars(t) - env.free_vars()))
    return Scheme(quantified, t)


# --------------------------------------------------------------------------- #
# Algorithm W: infer a principal type for a term                              #
# --------------------------------------------------------------------------- #


def infer(env: TypeEnv, term: Term) -> tuple[Subst, Type]:
    if isinstance(term, Lit):
        return {}, TInt

    if isinstance(term, Var):
        if term.name not in env.env:
            raise TypeError_(f"unbound variable: {term.name}")
        return {}, instantiate(env.env[term.name])

    if isinstance(term, Lam):
        # Param gets a fresh monotype var (NOT generalized) while we check body.
        tv = fresh_var()
        body_env = env.extend(term.param, Scheme((), tv))
        s1, t_body = infer(body_env, term.body)
        return s1, TArrow(apply(s1, tv), t_body)

    if isinstance(term, App):
        # (fn arg): unify fn's type with (arg_type -> fresh result).
        s1, t_fn = infer(env, term.fn)
        s2, t_arg = infer(apply_env(s1, env), term.arg)
        tv = fresh_var()
        s3 = unify(apply(s2, t_fn), TArrow(t_arg, tv))
        return compose(s3, compose(s2, s1)), apply(s3, tv)

    if isinstance(term, Let):
        # Infer the value, GENERALIZE it, then check the body with the scheme.
        s1, t_val = infer(env, term.value)
        scheme = generalize(apply_env(s1, env), apply(s1, t_val))
        body_env = apply_env(s1, env).extend(term.name, scheme)
        s2, t_body = infer(body_env, term.body)
        return compose(s2, s1), t_body

    raise TypeError_(f"unknown term: {term!r}")


def apply_env(sub: Subst, env: TypeEnv) -> TypeEnv:
    return TypeEnv({k: apply_scheme(sub, v) for k, v in env.env.items()})


# --------------------------------------------------------------------------- #
# Pretty-printing: rename t17 -> a, b, c ... so principal types read cleanly   #
# --------------------------------------------------------------------------- #


def show(t: Type) -> str:
    if isinstance(t, TVar):
        return t.name
    if isinstance(t, TCon):
        return t.name
    # Parenthesize the argument if it is itself a function (right-assoc arrows).
    left = f"({show(t.arg)})" if isinstance(t.arg, TArrow) else show(t.arg)
    return f"{left} -> {show(t.ret)}"


def normalize(t: Type) -> Type:
    """Rename free vars to a, b, c, ... in order of appearance."""
    names: dict[str, str] = {}
    letters = (chr(ord("a") + i) for i in count())

    def go(ty: Type) -> Type:
        if isinstance(ty, TVar):
            if ty.name not in names:
                names[ty.name] = next(letters)
            return TVar(names[ty.name])
        if isinstance(ty, TArrow):
            return TArrow(go(ty.arg), go(ty.ret))
        return ty

    return go(t)


def principal_type(term: Term) -> str:
    _, t = infer(TypeEnv(), term)
    return show(normalize(t))


# --------------------------------------------------------------------------- #
# Demo: infer principal types for classic terms, with NO annotations anywhere #
# --------------------------------------------------------------------------- #


def main() -> None:
    # \x. x                         the identity function
    identity = Lam("x", Var("x"))

    # \x. \y. x                     const / K combinator
    const = Lam("x", Lam("y", Var("x")))

    # \f. \x. f (f x)               apply f twice
    twice = Lam("f", Lam("x", App(Var("f"), App(Var("f"), Var("x")))))

    # \f. \g. \x. f (g x)           function composition
    compose_term = Lam(
        "f",
        Lam("g", Lam("x", App(Var("f"), App(Var("g"), Var("x"))))),
    )

    # let id = \x. x in id id       polymorphic use: id at two different types
    let_id_id = Let("id", identity, App(Var("id"), Var("id")))

    # (\x. x) 42                    apply identity to an int literal
    apply_lit = App(identity, Lit(42))

    examples: list[tuple[str, Term]] = [
        ("\\x. x", identity),
        ("\\x. \\y. x", const),
        ("\\f. \\x. f (f x)", twice),
        ("\\f. \\g. \\x. f (g x)", compose_term),
        ("let id = \\x. x in id id", let_id_id),
        ("(\\x. x) 42", apply_lit),
    ]

    print("Hindley-Milner principal types (inferred, zero annotations):\n")
    width = max(len(src) for src, _ in examples)
    for src, term in examples:
        print(f"  {src:<{width}}  ::  {principal_type(term)}")

    # The occurs check in action: \x. x x has no finite type (self-application).
    self_app = Lam("x", App(Var("x"), Var("x")))
    print("\nThe occurs check rejects infinite types:")
    try:
        principal_type(self_app)
    except TypeError_ as exc:
        print(f"  \\x. x x  ::  REJECTED -- {exc}")

    print("\nOK")


if __name__ == "__main__":
    main()
