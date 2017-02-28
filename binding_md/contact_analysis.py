#!/usr/bin/env python

import itertools
import collections
import numpy as np
import mdtraj as md
from contact_map import ContactFrequency
from contact_map import concurrence


"""
Tools for analyzing binding processes.


"""

def pick_window(min_dists, cutoff=0.45, cutoff_min=0.0, duration=20.0,
                padding=1.0, dt=0.05):
    """Find subtrajectories that seem to be attached.

    Parameters
    ----------
    min_dists : np.array
        minimum distance at each time
    cutoff : float
        distance to count as a contact (nm)
    duration : float
        duration to stay below cutoff to be considered stable (ns)
    padding : float
        time (ns) to skip analysis on either side of trajectory
    dt : float
        time (ns) between frames
    """
    contacted = [cutoff_min < d < cutoff for d in min_dists]
    ranges = []
    start = None
    for (i, val) in enumerate(contacted):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(contacted)))

    duration_frames = int(duration / dt)
    padding_frames = int(padding / dt)
    return [(r[0]+padding_frames, r[1]-padding_frames)
            for r in ranges if r[1] - r[0] > duration_frames]

class LigandContactAnalysis(object):
    def __init__(self, trajectory, ligand_groups, ligand_resids,
                 cutoff=0.45, freq_cutoff=0.2, contact_file=None):
        self.ligand_groups = ligand_groups
        self.ligand_atoms = sum(ligand_groups.values(), [])
        self.topology = trajectory.topology
        self.ligand_resids = ligand_resids
        self.cutoff = cutoff
        self.freq_cutoff = freq_cutoff
        self.n_frames = len(trajectory)

        # load the contacts
        protein_heavy = self.topology.select("protein and symbol != 'H'")
        if contact_file is None:
            self.contacts = ContactFrequency(trajectory,
                                             query=self.ligand_atoms,
                                             haystack=protein_heavy,
                                             cutoff=self.cutoff)
        else:
            try:
                self.contacts = ContactFrequency.from_file(contact_file)
            except:
                self.contacts = ContactFrequency(trajectory,
                                                 query=self.ligand_atoms,
                                                 haystack=protein_heavy,
                                                 cutoff=self.cutoff)

        # create the concurrence info
        per_residue_heavy = self._per_residue_heavy()
        self.group_minimum_distances = self._group_minimum_distances(
            trajectory,
            query_groups=self.ligand_groups,
            haystack_groups=per_residue_heavy
        )

    def trajectory_length(self, dt=1):
        return self.n_frames * dt

    def _per_residue_heavy(self):
        all_residues_in_contacts = set(
            sum([c[0]
                 for c in self.contacts.residue_contacts.most_common()], [])
        )
        ligand_residue_strings = [str(self.topology.residue(rr))
                                  for rr in self.ligand_resids]
        residues_in_contact = [
            r for r in all_residues_in_contacts
            if str(r) not in ligand_residue_strings
        ]
        return {r: [a.index for a in r.atoms if a.element.symbol != "H"]
                for r in residues_in_contact}

    def _group_minimum_distances(self, trajectory, query_groups,
                                 haystack_groups):
        group_minimum_distances = {}
        iterator = itertools.product(query_groups.keys(),
                                     haystack_groups.keys())
        for query_k, haystack_k in iterator:
            query = query_groups[query_k]
            haystack = haystack_groups[haystack_k]
            atom_pairs = list(itertools.product(query, haystack))
            min_dists = md.compute_distances(
                trajectory, atom_pairs=atom_pairs
            ).min(axis=1)
            group_minimum_distances[(query_k, haystack_k)] = min_dists

        return group_minimum_distances


    def ligand_group_contact_frequency(self, freq_cutoff=None):
        if freq_cutoff is None:
            freq_cutoff = self.freq_cutoff
        group_contact_frequency = {}
        for (label, min_dists) in self.group_minimum_distances.items():
            count = sum([d < self.cutoff for d in min_dists])
            freq = float(count) / self.n_frames
            if freq > freq_cutoff:
                group_contact_frequency[label] = freq

        return collections.Counter(group_contact_frequency)

    def ligand_group_concurrence(self, n_contacts=15, freq_cutoff=None):
        contact_count = self.ligand_group_contact_frequency(freq_cutoff)
        contacts_to_use = contact_count.most_common()[:n_contacts]
        values = []
        labels = []
        for contact in contacts_to_use:
            contact_label = contact[0]
            min_dists = self.group_minimum_distances[contact_label]
            values.append(map(lambda d: d < self.cutoff, min_dists))
            labels.append(contact_label)
        return concurrence.Concurrence(values, labels)

    @property
    def contact_minimum_distance(self):
        all_min_dist = np.stack(self.group_minimum_distances.values())
        return all_min_dist.min(axis=0)

    def top_n_contacts_for_ligand_groups(self, n_contacts, freq_cutoff=None):
        if freq_cutoff is None:
            freq_cutoff = self.freq_cutoff
        contact_freq = self.ligand_group_contact_frequency(freq_cutoff)
        most_common = contact_freq.most_common()
        results = collections.defaultdict(list)
        for contact in most_common:
            ligand_group = contact[0][0]
            residue = contact[0][1]
            if len(results[ligand_group]) < n_contacts:
                results[ligand_group] += [residue]

        return results

