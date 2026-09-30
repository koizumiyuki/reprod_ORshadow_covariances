#!/usr/bin/env python3
"""Exact covariance coefficients for orbital-shadow RDM estimators.

This script evaluates low-order coefficients using the estimator and
Grassmannian-moment framework of Koizumi et al., arXiv:2606.30601v1:

* Grassmannian coordinate moments from the convolution formula.
* The projector Weingarten function from the character formula.
* Sector coefficients A_W and covariance coefficients Gamma_K.
* Reference tests and consistency checks.

Run:

    python fs_covariance.py --write-tex --checks

The generated TeX files are written below ``outputs/``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Iterable

import sympy as sp


N, eta = sp.symbols("N eta")

MatrixEntry = tuple[int, int]
Monomial = tuple[MatrixEntry, ...]
Poly = dict[Monomial, sp.Expr]


def rat(expr: sp.Expr) -> sp.Expr:
    """Canonical rational simplification for this problem."""
    return sp.factor(sp.together(expr))


def cancel_rational(expr: sp.Expr) -> sp.Expr:
    """Lighter rational simplification for cached intermediate values."""
    return sp.together(expr)


def poly_clean(poly: Poly) -> Poly:
    out: Poly = {}
    for monomial, coeff in poly.items():
        if coeff != 0:
            out[monomial] = coeff
    return out


def poly_add(*polys: Poly) -> Poly:
    out: dict[Monomial, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for poly in polys:
        for monomial, coeff in poly.items():
            out[monomial] += coeff
    return poly_clean(dict(out))


def poly_scale(poly: Poly, scalar: sp.Expr) -> Poly:
    if scalar == 0:
        return {}
    return poly_clean({monomial: scalar * coeff for monomial, coeff in poly.items()})


def poly_mul(a: Poly, b: Poly) -> Poly:
    out: dict[Monomial, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for ma, ca in a.items():
        for mb, cb in b.items():
            out[tuple(sorted(ma + mb))] += ca * cb
    return poly_clean(dict(out))


def poly_entry(i: int, j: int, coeff: sp.Expr = sp.Integer(1)) -> Poly:
    return {((i, j),): coeff}


def poly_const(coeff: sp.Expr) -> Poly:
    return {(): coeff} if coeff != 0 else {}


def delta(a: int, b: int) -> int:
    return 1 if a == b else 0


def perm_sign(perm: tuple[int, ...]) -> int:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            inv += int(perm[i] > perm[j])
    return -1 if inv % 2 else 1


def powerset(items: Iterable[int]) -> list[tuple[int, ...]]:
    xs = tuple(sorted(items))
    out: list[tuple[int, ...]] = []
    for r in range(len(xs) + 1):
        out.extend(itertools.combinations(xs, r))
    return out


def det_poly(rows: Iterable[int], cols: Iterable[int]) -> Poly:
    rows = tuple(sorted(rows))
    cols = tuple(sorted(cols))
    if len(rows) != len(cols):
        raise ValueError("determinant needs equally many rows and columns")
    d = len(rows)
    if d == 0:
        return poly_const(sp.Integer(1))

    terms: list[Poly] = []
    for perm in itertools.permutations(range(d)):
        term = poly_const(sp.Integer(perm_sign(perm)))
        for a, b in enumerate(perm):
            term = poly_mul(term, poly_entry(rows[a], cols[b]))
        terms.append(term)
    return poly_add(*terms)


def estimator_poly(left: tuple[int, ...], right: tuple[int, ...]) -> Poly:
    """Estimator entry Dhat_{left,right}; tuple length is 1 or 2."""
    if len(left) != len(right):
        raise ValueError("estimator tuples must have the same length")
    if len(left) == 1:
        i, j = left[0], right[0]
        return poly_add(
            poly_entry(i, j, N + 1),
            poly_const(-eta * delta(i, j)),
        )
    if len(left) == 2:
        a1, a2 = left
        b1, b2 = right
        c2 = N * (N + 1) / sp.Integer(2)
        c1 = -(N + 1) * (eta - 1) / sp.Integer(2)
        out = poly_scale(det_poly(left, right), c2)
        linear = poly_add(
            poly_entry(a1, b1, delta(a2, b2)),
            poly_entry(a1, b2, -delta(a2, b1)),
            poly_entry(a2, b1, -delta(a1, b2)),
            poly_entry(a2, b2, delta(a1, b1)),
        )
        out = poly_add(out, poly_scale(linear, c1))
        if left == right:
            out = poly_add(out, poly_const(eta * (eta - 1) / sp.Integer(2)))
        return out
    raise ValueError("only one- and two-body estimators are implemented")


@dataclass(frozen=True)
class Profile:
    L: tuple[int, ...]
    L_plus: tuple[int, ...]
    L_minus: tuple[int, ...]
    L0: tuple[int, ...]
    m: int
    impossible: bool
    zeta: dict[int, int]


@dataclass(frozen=True)
class Quadruple:
    p: tuple[int, ...]
    q: tuple[int, ...]
    r: tuple[int, ...]
    s: tuple[int, ...]
    family: str
    name: str
    label: str

    def display(self) -> str:
        if self.family == "k1":
            return f"({self.p[0]},{self.q[0]};{self.r[0]},{self.s[0]})"
        if self.family == "mixed":
            return f"({self.p[0]};{self.q[0]};{pair_text(self.r)},{pair_text(self.s)})"
        return f"({pair_text(self.p)},{pair_text(self.q)};{pair_text(self.r)},{pair_text(self.s)})"

    def hermitian_partner(self) -> "Quadruple":
        return Quadruple(self.q, self.p, self.s, self.r, self.family, self.name, self.label)


def pair_text(pair: tuple[int, ...]) -> str:
    return "".join(str(x) for x in pair)


def charge_profile(p: tuple[int, ...], q: tuple[int, ...], r: tuple[int, ...], s: tuple[int, ...]) -> Profile:
    L = tuple(sorted(set(p) | set(q) | set(r) | set(s)))
    zeta: dict[int, int] = {}
    for x in L:
        z = int(x in p) - int(x in q) + int(x in s) - int(x in r)
        zeta[x] = z
    impossible = any(abs(z) == 2 for z in zeta.values())
    L_plus = tuple(x for x in L if zeta[x] == 1)
    L_minus = tuple(x for x in L if zeta[x] == -1)
    L0 = tuple(x for x in L if zeta[x] == 0)
    return Profile(L, L_plus, L_minus, L0, len(L_plus), impossible, zeta)


def merge_sign(P: Iterable[int], M: Iterable[int], C: Iterable[int]) -> int:
    P = tuple(sorted(P))
    M_desc = tuple(sorted(M, reverse=True))
    C = tuple(sorted(C))

    s1: list[tuple[str, int]] = []
    s1.extend(("create", y) for y in P)
    s1.extend(("ann", x) for x in M_desc)
    for c in C:
        s1.append(("create", c))
        s1.append(("ann", c))

    s2: list[tuple[str, int]] = []
    s2.extend(("create", y) for y in sorted(set(P) | set(C)))
    s2.extend(("ann", x) for x in sorted(set(M) | set(C), reverse=True))

    pos = {sym: i for i, sym in enumerate(s2)}
    positions = [pos[sym] for sym in s1]
    return perm_sign(tuple(positions))


def trace_factor_poly(profile: Profile, W: tuple[int, ...]) -> Poly:
    W_set = set(W)
    terms: list[Poly] = []
    for U in powerset(set(profile.L0) - W_set):
        C = tuple(sorted(W_set | set(U)))
        sign = (-1) ** len(U) * merge_sign(profile.L_plus, profile.L_minus, C)
        rows = tuple(sorted(set(profile.L_minus) | set(C)))
        cols = tuple(sorted(set(profile.L_plus) | set(C)))
        terms.append(poly_scale(det_poly(rows, cols), sp.Integer(sign)))
    return poly_add(*terms)


def prefactor(profile: Profile, W: tuple[int, ...]) -> sp.Expr:
    """binom(N,eta) / binom(N-|L|, eta-m-|W|) as a rational product."""
    ell = len(profile.L)
    r = profile.m + len(W)
    numerator = sp.prod(N - t for t in range(ell))
    eta_den = sp.prod(eta - t for t in range(r))
    hole_den = sp.prod(N - eta - t for t in range(ell - r))
    den = eta_den * hole_den
    return rat(numerator / den)


def partition_tuples(n: int, max_part: int | None = None) -> tuple[tuple[int, ...], ...]:
    if n == 0:
        return ((),)
    if max_part is None or max_part > n:
        max_part = n
    out: list[tuple[int, ...]] = []
    for first in range(max_part, 0, -1):
        for tail in partition_tuples(n - first, min(first, n - first) if n - first else 0):
            out.append((first,) + tail)
    return tuple(out)


def hook_dimension(lam: tuple[int, ...]) -> int:
    n = sum(lam)
    if n == 0:
        return 1
    hooks: list[int] = []
    for r, row_len in enumerate(lam):
        for c in range(row_len):
            right = row_len - c - 1
            below = sum(1 for rr in range(r + 1, len(lam)) if lam[rr] > c)
            hooks.append(right + below + 1)
    return math.factorial(n) // math.prod(hooks)


def boxes(lam: tuple[int, ...]) -> set[tuple[int, int]]:
    return {(r, c) for r, row_len in enumerate(lam) for c in range(row_len)}


def normalize_partition(parts: Iterable[int]) -> tuple[int, ...]:
    out = tuple(int(x) for x in parts)
    while out and out[-1] == 0:
        out = out[:-1]
    return out


def subpartitions_of_size(lam: tuple[int, ...], size: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(row: int, prev: int, remaining: int, acc: list[int]) -> None:
        if row == len(lam):
            if remaining == 0:
                out.append(normalize_partition(acc))
            return
        max_len = min(lam[row], prev, remaining)
        for val in range(max_len, -1, -1):
            if remaining - val < 0:
                continue
            rec(row + 1, val, remaining - val, acc + [val])

    rec(0, lam[0] if lam else 0, size, [])
    return sorted(set(out), reverse=True)


def is_connected(shape: set[tuple[int, int]]) -> bool:
    if not shape:
        return False
    start = next(iter(shape))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in shape and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(shape)


def has_no_two_by_two(shape: set[tuple[int, int]]) -> bool:
    for r, c in shape:
        if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= shape:
            return False
    return True


@cache
def rim_hooks(lam: tuple[int, ...], size: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    target_size = sum(lam) - size
    if target_size < 0:
        return ()
    lam_boxes = boxes(lam)
    out: list[tuple[tuple[int, ...], int]] = []
    for mu in subpartitions_of_size(lam, target_size):
        strip = lam_boxes - boxes(mu)
        if len(strip) != size:
            continue
        if is_connected(strip) and has_no_two_by_two(strip):
            height = len({r for r, _ in strip})
            out.append((mu, height))
    return tuple(out)


@cache
def character(lam: tuple[int, ...], mu: tuple[int, ...]) -> int:
    lam = normalize_partition(lam)
    mu = tuple(sorted((x for x in mu if x > 0), reverse=True))
    if sum(lam) != sum(mu):
        return 0
    if not mu:
        return 1 if not lam else 0
    r = mu[0]
    total = 0
    for remaining, height in rim_hooks(lam, r):
        total += (-1) ** (height - 1) * character(remaining, mu[1:])
    return total


def cycle_type(sigma: tuple[int, ...]) -> tuple[int, ...]:
    d = len(sigma)
    seen = [False] * d
    cycles: list[int] = []
    for start in range(d):
        if seen[start]:
            continue
        cur = start
        length = 0
        while not seen[cur]:
            seen[cur] = True
            length += 1
            cur = sigma[cur]
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))


@cache
def wg(mu: tuple[int, ...]) -> sp.Expr:
    d = sum(mu)
    if d == 0:
        return sp.Integer(1)
    total = sp.Integer(0)
    for lam in partition_tuples(d):
        dim = hook_dimension(lam)
        chi = character(lam, mu)
        if chi == 0:
            continue
        content_product = sp.Integer(1)
        for row, row_len in enumerate(lam):
            for col in range(row_len):
                content = col - row
                content_product *= (eta + content) / (N + content)
        total += dim * chi * content_product
    return rat(total / math.factorial(d))


@cache
def monomial_type_counts(monomial: Monomial) -> tuple[tuple[tuple[int, ...], int], ...]:
    d = len(monomial)
    if d == 0:
        return (((), 1),)
    i_seq = tuple(i for i, _ in monomial)
    j_seq = tuple(j for _, j in monomial)
    if Counter(i_seq) != Counter(j_seq):
        return ()

    groups: list[tuple[list[int], list[tuple[int, ...]]]] = []
    for value in sorted(set(j_seq)):
        targets = [a for a, j in enumerate(j_seq) if j == value]
        sources = [a for a, i in enumerate(i_seq) if i == value]
        groups.append((targets, list(itertools.permutations(sources))))

    type_counts: Counter[tuple[int, ...]] = Counter()
    for assignments in itertools.product(*(perms for _, perms in groups)):
        sigma = [None] * d
        for (targets, _), selected_sources in zip(groups, assignments):
            for target, source in zip(targets, selected_sources):
                sigma[target] = source
        type_counts[cycle_type(tuple(int(x) for x in sigma))] += 1

    return tuple(sorted(type_counts.items()))


@cache
def monomial_moment(monomial: Monomial) -> sp.Expr:
    counts = monomial_type_counts(monomial)
    if not counts:
        return sp.Integer(0)
    if len(counts) == 1:
        mu, count = counts[0]
        return count * wg(mu)
    return cancel_rational(sum(count * wg(mu) for mu, count in counts))


@cache
def sector_coefficient_cached(p: tuple[int, ...], q: tuple[int, ...], r: tuple[int, ...], s: tuple[int, ...], W: tuple[int, ...]) -> sp.Expr:
    profile = charge_profile(p, q, r, s)
    if profile.impossible:
        return sp.Integer(0)
    f = poly_mul(estimator_poly(p, q), estimator_poly(s, r))
    t = trace_factor_poly(profile, W)
    integrand = poly_mul(f, t)
    wg_coeffs: dict[tuple[int, ...], sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for monomial, coeff in integrand.items():
        for mu, count in monomial_type_counts(monomial):
            wg_coeffs[mu] += coeff * count
    integral = sp.Integer(0)
    for mu, coeff in wg_coeffs.items():
        integral += coeff * wg(mu)
    return rat(prefactor(profile, W) * integral)


def sector_coefficient(case: Quadruple, W: tuple[int, ...]) -> sp.Expr:
    return sector_coefficient_cached(case.p, case.q, case.r, case.s, tuple(sorted(W)))


@cache
def coefficients_cached(p: tuple[int, ...], q: tuple[int, ...], r: tuple[int, ...], s: tuple[int, ...]) -> tuple[dict[tuple[int, ...], sp.Expr], dict[tuple[int, ...], sp.Expr]]:
    profile = charge_profile(p, q, r, s)
    if profile.impossible:
        return {}, {}
    A: dict[tuple[int, ...], sp.Expr] = {}
    Gamma: dict[tuple[int, ...], sp.Expr] = {}
    for W in powerset(profile.L0):
        A[W] = sector_coefficient_cached(p, q, r, s, W)
    for K in powerset(profile.L0):
        gamma = sp.Integer(0)
        for W in powerset(K):
            gamma += (-1) ** (len(K) - len(W)) * A[W]
        Gamma[K] = rat(gamma)
    return A, Gamma


def coefficients(case: Quadruple) -> tuple[dict[tuple[int, ...], sp.Expr], dict[tuple[int, ...], sp.Expr]]:
    return coefficients_cached(case.p, case.q, case.r, case.s)


def make_cases() -> list[Quadruple]:
    cases: list[Quadruple] = []

    def q1(name: str, rep: tuple[int, int, int, int]) -> None:
        i, j, k, l = rep
        cases.append(Quadruple((i,), (j,), (k,), (l,), "k1", name, f"$k=1$, class {name}"))

    q1("I", (1, 2, 2, 1))
    q1("II", (1, 1, 1, 1))
    q1("III", (1, 1, 2, 2))
    q1("III'", (1, 2, 1, 2))
    q1("IV", (1, 2, 1, 1))
    q1("V", (1, 2, 1, 3))
    q1("V'", (1, 1, 2, 3))
    q1("VI", (1, 2, 3, 4))

    def pair(text: str) -> tuple[int, int]:
        return (int(text[0]), int(text[1]))

    def q2(name: str, rep: tuple[str, str, str, str]) -> None:
        p, q, r, s = (pair(x) for x in rep)
        cases.append(Quadruple(p, q, r, s, "k2", name, f"$k=2$, class {name}"))

    q2("I", ("12", "34", "56", "17"))
    q2("II", ("12", "12", "12", "12"))
    q2("III", ("12", "12", "13", "13"))
    q2("III'", ("12", "13", "12", "13"))
    q2("IV", ("12", "12", "34", "34"))
    q2("IV'", ("12", "13", "24", "34"))
    q2("IV''", ("12", "34", "12", "34"))
    q2("V", ("12", "12", "12", "13"))
    q2("V'", ("12", "12", "13", "23"))
    q2("V''", ("12", "13", "12", "23"))
    q2("VI", ("12", "12", "13", "14"))
    q2("VI'", ("12", "12", "13", "34"))
    q2("VI''", ("12", "13", "12", "14"))
    q2("VI'''", ("12", "13", "12", "34"))
    q2("VI''''", ("12", "13", "14", "34"))
    q2("VII", ("12", "13", "24", "45"))
    q2("VII'", ("12", "12", "34", "35"))
    q2("VII''", ("12", "13", "24", "35"))
    q2("VII'''", ("12", "34", "12", "35"))
    q2("VIII", ("12", "34", "13", "13"))
    q2("VIII'", ("12", "12", "12", "34"))
    q2("VIII''", ("12", "13", "12", "24"))
    q2("VIII'''", ("12", "13", "13", "34"))
    q2("IX", ("12", "13", "14", "15"))
    q2("IX'", ("12", "12", "13", "45"))
    q2("IX''", ("12", "13", "12", "45"))
    q2("IX'''", ("12", "13", "14", "35"))
    q2("IX''''", ("12", "13", "14", "45"))
    q2("X", ("12", "13", "45", "46"))
    q2("X'", ("12", "12", "34", "56"))
    q2("X''", ("12", "13", "24", "56"))
    q2("X'''", ("12", "34", "12", "56"))
    q2("X''''", ("12", "34", "15", "36"))
    q2("XI", ("12", "45", "16", "13"))
    q2("XII", ("12", "45", "67", "37"))
    q2("XII'", ("12", "34", "15", "67"))
    q2("XIII", ("12", "34", "78", "56"))

    def qm(name: str, rep: tuple[int, int, str, str]) -> None:
        i, j, r, s = rep
        cases.append(Quadruple((i,), (j,), pair(r), pair(s), "mixed", name, f"mixed $(1,2)$, class {name}"))

    qm("I", (1, 2, "12", "13"))
    qm("C1", (1, 1, "12", "12"))
    qm("C2", (1, 1, "23", "23"))
    qm("C3", (1, 2, "13", "23"))
    qm("C4", (1, 2, "12", "12"))
    qm("C5", (1, 1, "12", "13"))
    qm("C6", (1, 1, "12", "23"))
    qm("C7", (1, 2, "12", "23"))
    qm("C8", (1, 2, "13", "13"))
    qm("C9", (1, 1, "23", "24"))
    qm("C10", (1, 2, "13", "24"))
    qm("C11", (1, 2, "13", "34"))
    qm("C12", (1, 2, "34", "34"))
    qm("C13", (1, 1, "12", "34"))
    qm("C14", (1, 2, "13", "14"))
    qm("C15", (1, 1, "23", "45"))
    qm("C16", (1, 2, "13", "45"))
    qm("C17", (1, 2, "34", "35"))
    qm("C18", (1, 2, "34", "56"))
    return cases


def set_tex(xs: tuple[int, ...]) -> str:
    if not xs:
        return r"\emptyset"
    return r"\{" + ",".join(str(x) for x in xs) + r"\}"


def expr_tex(expr: sp.Expr) -> str:
    return sp.latex(rat(expr))


def render_coeff_block(case: Quadruple, A: dict[tuple[int, ...], sp.Expr], Gamma: dict[tuple[int, ...], sp.Expr]) -> str:
    profile = charge_profile(case.p, case.q, case.r, case.s)
    if profile.impossible:
        return (
            f"\\paragraph{{{case.label}, ${case.display()}$}}\n"
            "Class I: impossible charge, so no coefficients are defined.\n\n"
        )

    lines: list[str] = [
        f"\\paragraph{{{case.label}, ${case.display()}$}}\n",
        f"$L_+={set_tex(profile.L_plus)}$, $L_-={set_tex(profile.L_minus)}$, $L_0={set_tex(profile.L0)}$.\n",
        r"\begin{align*}",
    ]

    ordered = powerset(profile.L0)
    for idx, W in enumerate(ordered):
        suffix = r"\\" if idx < len(ordered) - 1 else ""
        lines.append(f"  A_{{{set_tex(W)}}} &= {expr_tex(A[W])} {suffix}")
    lines.append(r"\end{align*}")
    lines.append(r"\begin{align*}")
    for idx, K in enumerate(ordered):
        suffix = r"\\" if idx < len(ordered) - 1 else ""
        lines.append(f"  \\Gamma_{{{set_tex(K)}}} &= {expr_tex(Gamma[K])} {suffix}")
    lines.append(r"\end{align*}")
    lines.append("")
    return "\n".join(lines)


def reference_cases(cases: list[Quadruple]) -> list[Quadruple]:
    keys = {("k1", "V"), ("k2", "VI'''"), ("mixed", "C5")}
    return [case for case in cases if (case.family, case.name) in keys]


def render_tex(cases: list[Quadruple], progress: bool = False) -> str:
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=2.3cm]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\allowdisplaybreaks",
        r"\begin{document}",
        r"\section*{Orbital-shadow covariance coefficients}",
        "",
    ]
    for idx, case in enumerate(cases, start=1):
        if progress:
            print(f"[tex] {idx}/{len(cases)} {case.family} {case.name} {case.display()}", file=sys.stderr, flush=True)
        A, Gamma = coefficients(case)
        parts.append(render_coeff_block(case, A, Gamma))
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def expected_reference() -> dict[tuple[str, str, tuple[int, ...], str], sp.Expr]:
    return {
        ("k1", "V", (), "A"): (N + 1) * (N - eta + 1) / (N * (N + 2)),
        ("k1", "V", (1,), "A"): -(N + 1) * (eta + 1) / (N * (N + 2)),
        ("k1", "V", (), "G"): (N + 1) * (N - eta + 1) / (N * (N + 2)),
        ("k1", "V", (1,), "G"): -(N + 1) / N,
        ("k2", "VI'''", (), "A"): -((eta - 1) * (N - eta + 1) * (2 * N - eta + 1)) / (4 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (2,), "A"): -((eta + 1) * (N - eta + 1) ** 2) / (4 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (3,), "A"): -((eta + 1) * (N - eta + 1) * (2 * N - eta + 1)) / (4 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (2, 3), "A"): -((eta + 1) * (N - eta - 1) * (N - eta + 1)) / (4 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (), "G"): -((eta - 1) * (N - eta + 1) * (2 * N - eta + 1)) / (4 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (2,), "G"): ((N - eta + 1) * (N * eta - 3 * N + 2 * eta - 2)) / (4 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (3,), "G"): -((N - eta + 1) * (2 * N - eta + 1)) / (2 * (N - 1) * (N + 2)),
        ("k2", "VI'''", (2, 3), "G"): ((N + 1) * (N - eta + 1)) / ((N - 1) * (N + 2)),
        ("mixed", "C5", (), "A"): (N + 1) * (eta - 1) * (N - eta + 1) / (2 * N * (N + 2)),
        ("mixed", "C5", (1,), "A"): (N - eta + 1) * (N * eta + 3 * N + eta + 1) / (2 * N * (N + 2)),
        ("mixed", "C5", (), "G"): (N + 1) * (eta - 1) * (N - eta + 1) / (2 * N * (N + 2)),
        ("mixed", "C5", (1,), "G"): (2 * N + 1) * (N - eta + 1) / (N * (N + 2)),
    }


def check_reference(cases: list[Quadruple]) -> list[str]:
    failures: list[str] = []
    by_key = {(case.family, case.name): case for case in cases}
    expected = expected_reference()
    for (family, name, subset, kind), wanted in expected.items():
        A, Gamma = coefficients(by_key[(family, name)])
        got = A[subset] if kind == "A" else Gamma[subset]
        if rat(got - wanted) != 0:
            failures.append(f"reference mismatch {family} {name} {kind}_{subset}: got {got}, expected {wanted}")
    return failures


def check_hole_counting(cases: list[Quadruple]) -> list[str]:
    failures: list[str] = []
    for case in cases:
        profile = charge_profile(case.p, case.q, case.r, case.s)
        if profile.impossible:
            continue
        A, _ = coefficients(case)
        for W, expr in A.items():
            max_j = len(profile.L) - profile.m - len(W)
            cancelled = sp.cancel(sp.together(expr))
            _, den = sp.fraction(cancelled)
            for j in range(1, max_j):
                value = sp.factor(den.subs(eta, N - j))
                if value == 0:
                    failures.append(f"hole pole {case.family} {case.name} W={W} at eta=N-{j}")
    return failures


def basis_states(n_modes: int, n_particles: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(1, n_modes + 1), n_particles))


def apply_create(state: tuple[int, ...], mode: int) -> tuple[int, tuple[int, ...]] | None:
    if mode in state:
        return None
    smaller = sum(1 for x in state if x < mode)
    sign = -1 if smaller % 2 else 1
    return sign, tuple(sorted(state + (mode,)))


def apply_annihilate(state: tuple[int, ...], mode: int) -> tuple[int, tuple[int, ...]] | None:
    if mode not in state:
        return None
    smaller = sum(1 for x in state if x < mode)
    sign = -1 if smaller % 2 else 1
    return sign, tuple(x for x in state if x != mode)


def apply_operator(state: tuple[int, ...], symbols: list[tuple[str, int]]) -> tuple[int, tuple[int, ...]] | None:
    coeff = 1
    cur = state
    for kind, mode in reversed(symbols):
        if kind == "create":
            ans = apply_create(cur, mode)
        else:
            ans = apply_annihilate(cur, mode)
        if ans is None:
            return None
        sign, cur = ans
        coeff *= sign
    return coeff, cur


def expectation_symbols(psi: dict[tuple[int, ...], complex], symbols: list[tuple[str, int]]) -> complex:
    total = 0j
    for state, amp in psi.items():
        ans = apply_operator(state, symbols)
        if ans is None:
            continue
        coeff, final = ans
        total += complex(psi.get(final, 0)).conjugate() * coeff * amp
    return total


def rdm_value(psi: dict[tuple[int, ...], complex], A: Iterable[int], B: Iterable[int]) -> complex:
    A = tuple(sorted(A))
    B = tuple(sorted(B))
    symbols: list[tuple[str, int]] = []
    symbols.extend(("create", x) for x in B)
    symbols.extend(("ann", x) for x in sorted(A, reverse=True))
    return expectation_symbols(psi, symbols)


def random_state(n_modes: int, n_particles: int, seed: int) -> dict[tuple[int, ...], complex]:
    rng = random.Random(seed)
    basis = basis_states(n_modes, n_particles)
    raw = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in basis]
    norm = math.sqrt(sum(abs(z) ** 2 for z in raw))
    return {state: amp / norm for state, amp in zip(basis, raw)}


def covariance_from_gamma(case: Quadruple, psi: dict[tuple[int, ...], complex], n_value: int, eta_value: int) -> complex:
    profile = charge_profile(case.p, case.q, case.r, case.s)
    if profile.impossible:
        return -rdm_value(psi, case.p, case.q) * rdm_value(psi, case.s, case.r)
    _, Gamma = coefficients(case)
    total = 0j
    for K, gamma_expr in Gamma.items():
        if profile.m + len(K) > eta_value:
            continue
        sign = merge_sign(profile.L_minus, profile.L_plus, K)
        gamma_num = complex(gamma_expr.subs({N: n_value, eta: eta_value}).evalf(50))
        A_set = tuple(sorted(set(profile.L_plus) | set(K)))
        B_set = tuple(sorted(set(profile.L_minus) | set(K)))
        total += sign * gamma_num * rdm_value(psi, A_set, B_set)
    total -= rdm_value(psi, case.p, case.q) * rdm_value(psi, case.s, case.r)
    return total


def k1_closed(case: Quadruple, psi: dict[tuple[int, ...], complex], n_value: int, eta_value: int) -> complex:
    i, j, k, l = case.p[0], case.q[0], case.r[0], case.s[0]
    c1 = (n_value + 1 - eta_value) / (n_value * (n_value + 2))
    c2 = (n_value + 1) * c1
    four_symbols = [("create", j), ("create", k), ("ann", l), ("ann", i)]
    return (
        -c1
        * (
            eta_value * delta(i, j) * delta(k, l)
            + delta(i, j) * rdm_value(psi, (l,), (k,))
            + delta(k, l) * rdm_value(psi, (i,), (j,))
        )
        + c2
        * (
            eta_value * delta(i, k) * delta(j, l)
            + delta(i, k) * rdm_value(psi, (l,), (j,))
            + delta(j, l) * rdm_value(psi, (i,), (k,))
        )
        + ((n_value + 1) / n_value) * expectation_symbols(psi, four_symbols)
        - rdm_value(psi, case.p, case.q) * rdm_value(psi, case.s, case.r)
    )


def check_k1_anchor(cases: list[Quadruple]) -> list[str]:
    failures: list[str] = []
    n_value = 8
    eta_value = 3
    for seed in range(3):
        psi = random_state(n_value, eta_value, seed)
        for case in cases:
            if case.family != "k1":
                continue
            got = covariance_from_gamma(case, psi, n_value, eta_value)
            wanted = k1_closed(case, psi, n_value, eta_value)
            if abs(got - wanted) > 1e-9:
                failures.append(f"k1 anchor {case.name} seed={seed}: got {got}, expected {wanted}")
    return failures


def check_variance_positivity(cases: list[Quadruple]) -> list[str]:
    failures: list[str] = []
    variance_keys = {
        ("k1", "II"),
        ("k1", "III'"),
        ("k2", "II"),
        ("k2", "III'"),
        ("k2", "IV''"),
    }
    n_value = 8
    eta_value = 4
    for seed in range(5):
        psi = random_state(n_value, eta_value, 100 + seed)
        for case in cases:
            if (case.family, case.name) not in variance_keys:
                continue
            value = covariance_from_gamma(case, psi, n_value, eta_value)
            if abs(value.imag) > 1e-8:
                failures.append(f"variance complex {case.family} {case.name} seed={seed}: {value}")
            if value.real < -1e-8:
                failures.append(f"variance negative {case.family} {case.name} seed={seed}: {value.real}")
    return failures


def check_hermiticity(cases: list[Quadruple]) -> list[str]:
    failures: list[str] = []
    for case in cases:
        profile = charge_profile(case.p, case.q, case.r, case.s)
        if profile.impossible:
            continue
        A, Gamma = coefficients(case)
        partner = case.hermitian_partner()
        Ap, Gammap = coefficients(partner)
        for W, value in A.items():
            if rat(value - Ap[W]) != 0:
                failures.append(f"hermiticity A {case.family} {case.name} W={W}")
        for K, value in Gamma.items():
            if rat(value - Gammap[K]) != 0:
                failures.append(f"hermiticity Gamma {case.family} {case.name} K={K}")
    return failures


def run_checks(cases: list[Quadruple], selected: set[str] | None = None, progress: bool = False) -> str:
    sections = [
        ("reference test values", check_reference),
        ("hole counting", check_hole_counting),
        ("k=1 anchor", check_k1_anchor),
        ("variance positivity", check_variance_positivity),
        ("hermiticity", check_hermiticity),
    ]
    aliases = {
        "reference": "reference test values",
        "hole": "hole counting",
        "k1": "k=1 anchor",
        "variance": "variance positivity",
        "hermiticity": "hermiticity",
    }
    if selected:
        wanted = {aliases.get(name, name) for name in selected}
        sections = [(title, fn) for title, fn in sections if title in wanted]
    lines: list[str] = []
    all_failures: list[str] = []
    for title, fn in sections:
        if progress:
            print(f"[check] {title}", file=sys.stderr, flush=True)
        failures = fn(cases)
        if failures:
            lines.append(f"[FAIL] {title}: {len(failures)} failure(s)")
            lines.extend(f"  - {failure}" for failure in failures[:20])
            if len(failures) > 20:
                lines.append(f"  - ... {len(failures) - 20} more")
            all_failures.extend(failures)
        else:
            lines.append(f"[PASS] {title}")
    lines.append("")
    lines.append(f"overall: {'FAIL' if all_failures else 'PASS'}")
    return "\n".join(lines)


def write_outputs(outdir: Path, cases: list[Quadruple], progress: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reference_coefficients.tex").write_text(render_tex(reference_cases(cases), progress=progress), encoding="utf-8")
    (outdir / "all_coefficients.tex").write_text(render_tex(cases, progress=progress), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-tex", action="store_true", help="write outputs/reference_coefficients.tex and outputs/all_coefficients.tex")
    parser.add_argument("--checks", action="store_true", help="run all built-in consistency checks")
    parser.add_argument(
        "--check",
        action="append",
        choices=["reference", "hole", "k1", "variance", "hermiticity"],
        help="run only this named check; may be repeated",
    )
    parser.add_argument("--reference", action="store_true", help="print only the three reference TeX blocks")
    parser.add_argument("--all", action="store_true", help="print all TeX blocks to stdout")
    parser.add_argument("--outdir", default="outputs", help="output directory")
    parser.add_argument("--progress", action="store_true", help="print progress to stderr")
    args = parser.parse_args()

    cases = make_cases()
    checks_failed = False
    if args.write_tex:
        outdir = Path(args.outdir)
        write_outputs(outdir, cases, progress=args.progress)
    if args.checks or args.check:
        report = run_checks(cases, selected=set(args.check or []), progress=args.progress)
        print(report)
        checks_failed = report.endswith("overall: FAIL")
        if args.write_tex:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "check_report.txt").write_text(report, encoding="utf-8")
    if args.reference:
        print(render_tex(reference_cases(cases), progress=args.progress))
    if args.all:
        print(render_tex(cases, progress=args.progress))
    if not any((args.write_tex, args.checks, args.check, args.reference, args.all)):
        parser.print_help()
    if checks_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
